# ══════════════════════════════════════════════════════════════
# src/model.py — CustomYOLO + YOLOLossV2
# ══════════════════════════════════════════════════════════════

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import ANCHORS_NORM, LAMBDA_BOX, LAMBDA_OBJ, LAMBDA_CLS, LAMBDA_NOOBJ


# ── Modules de base
class Conv(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(oc)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, ic, oc, shortcut=True):
        super().__init__()
        self.cv1 = Conv(ic, oc, 1, p=0)
        self.cv2 = Conv(oc, oc)
        self.add = shortcut and ic == oc
    def forward(self, x): return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, ic, oc, n=1, shortcut=True):
        super().__init__()
        self.c   = int(oc*0.5)
        self.cv1 = Conv(ic, 2*self.c, 1, p=0)
        self.cv2 = Conv((2+n)*self.c, oc, 1, p=0)
        self.m   = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ── Backbone
class PretrainedBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        r = models.resnet50(weights=weights)
        print(f"  Backbone ResNet50 {'pré-entraîné' if pretrained else 'aléatoire'} chargé")
        self.stem   = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1;  self.layer2 = r.layer2
        self.layer3 = r.layer3;  self.layer4 = r.layer4
    def forward(self, x):
        x = self.stem(x);  x = self.layer1(x)
        p3 = self.layer2(x);  p4 = self.layer3(p3);  p5 = self.layer4(p4)
        return p3, p4, p5


# ── Neck FPN+PAN
class YOLOFPN(nn.Module):
    def __init__(self, bc=128):
        super().__init__()
        self.rp5 = Conv(2048, bc*4, 1, p=0);  self.rp4 = Conv(1024, bc*2, 1, p=0);  self.rp3 = Conv(512, bc, 1, p=0)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest');  self.c1 = C2f(bc*4+bc*2, bc*2)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest');  self.c2 = C2f(bc*2+bc,   bc)
        self.d1  = Conv(bc,   bc,   3, 2);  self.c3 = C2f(bc+bc*2,   bc*2)
        self.d2  = Conv(bc*2, bc*2, 3, 2);  self.c4 = C2f(bc*2+bc*4, bc*4)
    def forward(self, p3, p4, p5):
        p5=self.rp5(p5);  p4=self.rp4(p4);  p3=self.rp3(p3)
        fp4=self.c1(torch.cat([self.up1(p5),p4],1))
        fp3=self.c2(torch.cat([self.up2(fp4),p3],1))
        pp4=self.c3(torch.cat([self.d1(fp3),fp4],1))
        pp5=self.c4(torch.cat([self.d2(pp4),p5],1))
        return fp3, pp4, pp5


# ── Head
class DetectionHead(nn.Module):
    def __init__(self, nc=8, bc=128):
        super().__init__()
        self.no = nc+5
        self.h3 = nn.Conv2d(bc,    self.no*3, 1)
        self.h4 = nn.Conv2d(bc*2,  self.no*3, 1)
        self.h5 = nn.Conv2d(bc*4,  self.no*3, 1)
    def forward(self, p3, p4, p5): return [self.h3(p3), self.h4(p4), self.h5(p5)]


# ── Modèle complet
class CustomYOLO(nn.Module):
    def __init__(self, num_classes=8, base_channels=128, pretrained=True):
        super().__init__()
        print("Construction YOLO avec backbone pré-entraîné...")
        self.backbone    = PretrainedBackbone(pretrained=pretrained)
        self.neck        = YOLOFPN(base_channels)
        self.head        = DetectionHead(num_classes, base_channels)
        self.num_classes = num_classes
        self._init()
        print("  ✅ Backbone ResNet50 | ✅ Neck FPN/PAN | ✅ Head Detection")

    def _init(self):
        for m in [self.neck, self.head]:
            for l in m.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                    if l.bias is not None: nn.init.constant_(l.bias, 0)
                elif isinstance(l, nn.BatchNorm2d):
                    nn.init.constant_(l.weight, 1);  nn.init.constant_(l.bias, 0)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        print("Backbone gelé")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True
        print("Backbone dégelé")

    def forward(self, x):
        p3,p4,p5 = self.backbone(x)
        fp3,pp4,pp5 = self.neck(p3,p4,p5)
        return self.head(fp3,pp4,pp5)

    def summary(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}\n  YOLO : {total:,} params ({total*4/1024/1024:.1f} MB)\n{'='*60}")


# ── Loss
def bbox_iou_wh(wh1, wh2):
    inter = torch.min(wh1[0],wh2[0]) * torch.min(wh1[1],wh2[1])
    return inter / (wh1[0]*wh1[1] + wh2[0]*wh2[1] - inter + 1e-6)


class YOLOLossV2(nn.Module):
    def __init__(self, num_classes, img_size=640,
                 lambda_box=LAMBDA_BOX, lambda_obj=LAMBDA_OBJ,
                 lambda_cls=LAMBDA_CLS, lambda_noobj=LAMBDA_NOOBJ):
        super().__init__()
        self.nc = num_classes;  self.img_size = img_size
        self.lb = lambda_box;   self.lo = lambda_obj
        self.lc = lambda_cls;   self.ln = lambda_noobj
        self.anchors = ANCHORS_NORM
        self.bce     = nn.BCEWithLogitsLoss(reduction='none')

    def focal_bce(self, pred, target, gamma=2.0):
        bce = self.bce(pred, target)
        p_t = target*torch.sigmoid(pred) + (1-target)*(1-torch.sigmoid(pred))
        return bce * ((1-p_t)**gamma)

    def forward(self, predictions, targets):
        device = predictions[0].device
        losses = []

        for si, pred in enumerate(predictions):
            B,C,H,W = pred.shape
            anch    = self.anchors[si]
            at      = torch.tensor(anch, dtype=torch.float32, device=device)
            pred    = pred.view(B,3,5+self.nc,H,W).permute(0,1,3,4,2).contiguous()

            om = torch.zeros(B,3,H,W,device=device,dtype=torch.bool)
            nm = torch.ones (B,3,H,W,device=device,dtype=torch.bool)
            tx = torch.zeros(B,3,H,W,device=device);  ty=torch.zeros_like(tx)
            tw = torch.zeros_like(tx);                 th=torch.zeros_like(tx)
            tc = torch.zeros(B,3,H,W,self.nc,device=device)

            for b in range(B):
                if len(targets[b])==0: continue
                for t in targets[b].to(device):
                    ci=int(t[0]);  xc,yc,w,h=t[1].item(),t[2].item(),t[3].item(),t[4].item()
                    if w<=0 or h<=0: continue
                    wh=torch.tensor([w,h],device=device)
                    ba=int(torch.stack([bbox_iou_wh(wh,at[a]) for a in range(3)]).argmax())
                    gi=max(0,min(W-1,int(xc*W)));  gj=max(0,min(H-1,int(yc*H)))
                    om[b,ba,gj,gi]=True;  nm[b,ba,gj,gi]=False
                    tx[b,ba,gj,gi]=xc*W-gi;  ty[b,ba,gj,gi]=yc*H-gj
                    aw,ah=anch[ba]
                    tw[b,ba,gj,gi]=torch.log(torch.tensor(w/aw+1e-6,device=device))
                    th[b,ba,gj,gi]=torch.log(torch.tensor(h/ah+1e-6,device=device))
                    tc[b,ba,gj,gi,ci]=1.0

            if om.sum()>0:
                pxy=torch.sigmoid(pred[...,0:2]);  txy=torch.stack([tx,ty],dim=-1)
                lxy=F.mse_loss(pxy[om],txy[om])
                lwh=F.mse_loss(pred[...,2:4][om],torch.stack([tw,th],dim=-1)[om])
                lbox=lxy+lwh
                lcls=self.focal_bce(pred[...,5:][om],tc[om]).mean()
            else:
                lbox=lcls=torch.tensor(0.0,device=device)

            po=pred[...,4];  to=om.float()
            lop=self.focal_bce(po[om],to[om]).mean()   if om.sum()>0 else torch.tensor(0.0,device=device)
            lon=self.focal_bce(po[nm],to[nm]*0).mean() if nm.sum()>0 else torch.tensor(0.0,device=device)

            losses.append(self.lb*lbox + self.lo*lop + self.ln*lon + self.lc*lcls)

        return sum(losses)/3.0
