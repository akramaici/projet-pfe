# ══════════════════════════════════════════════════════════════
# src/model.py — CustomYOLO (ResNet50 + FPN/PAN) + YOLOLossV2
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


# ── Modules de base ───────────────────────────────────────────

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_ch,  out_ch, kernel_size=1, padding=0)
        self.conv2 = Conv(out_ch, out_ch, kernel_size=3, padding=1)
        self.add   = shortcut and in_ch == out_ch
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C2f(nn.Module):
    def __init__(self, in_ch, out_ch, n=1, shortcut=True):
        super().__init__()
        self.c   = int(out_ch * 0.5)
        self.cv1 = Conv(in_ch,          2*self.c, kernel_size=1, padding=0)
        self.cv2 = Conv((2+n)*self.c,   out_ch,   kernel_size=1, padding=0)
        self.m   = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ── Backbone ResNet50 ─────────────────────────────────────────

class PretrainedBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = models.resnet50(weights=weights)
        print(f"  Backbone ResNet50 {'pré-entraîné ImageNet' if pretrained else 'aléatoire'} chargé")
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2   # 512ch  → P3
        self.layer3 = resnet.layer3   # 1024ch → P4
        self.layer4 = resnet.layer4   # 2048ch → P5
    def forward(self, x):
        x  = self.stem(x);  x = self.layer1(x)
        p3 = self.layer2(x);  p4 = self.layer3(p3);  p5 = self.layer4(p4)
        return p3, p4, p5


# ── Neck FPN + PAN ────────────────────────────────────────────

class YOLOFPN(nn.Module):
    def __init__(self, base_channels=128):
        super().__init__()
        bc = base_channels
        self.reduce_p5   = Conv(2048, bc*4, kernel_size=1, padding=0)
        self.reduce_p4   = Conv(1024, bc*2, kernel_size=1, padding=0)
        self.reduce_p3   = Conv(512,  bc,   kernel_size=1, padding=0)
        self.upsample1   = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_up1     = C2f(bc*4+bc*2, bc*2, n=1)
        self.upsample2   = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_up2     = C2f(bc*2+bc,   bc,   n=1)
        self.downsample1 = Conv(bc,   bc,   kernel_size=3, stride=2, padding=1)
        self.c2f_down1   = C2f(bc+bc*2,   bc*2, n=1)
        self.downsample2 = Conv(bc*2, bc*2, kernel_size=3, stride=2, padding=1)
        self.c2f_down2   = C2f(bc*2+bc*4, bc*4, n=1)

    def forward(self, p3, p4, p5):
        p5 = self.reduce_p5(p5);  p4 = self.reduce_p4(p4);  p3 = self.reduce_p3(p3)
        fpn_p4 = self.c2f_up1(torch.cat([self.upsample1(p5), p4], 1))
        fpn_p3 = self.c2f_up2(torch.cat([self.upsample2(fpn_p4), p3], 1))
        pan_p4 = self.c2f_down1(torch.cat([self.downsample1(fpn_p3), fpn_p4], 1))
        pan_p5 = self.c2f_down2(torch.cat([self.downsample2(pan_p4), p5], 1))
        return fpn_p3, pan_p4, pan_p5


# ── Detection Head ────────────────────────────────────────────

class DetectionHead(nn.Module):
    def __init__(self, num_classes=8, base_channels=128):
        super().__init__()
        bc = base_channels;  self.no = num_classes + 5
        self.head_p3 = nn.Conv2d(bc,    self.no*3, 1)
        self.head_p4 = nn.Conv2d(bc*2,  self.no*3, 1)
        self.head_p5 = nn.Conv2d(bc*4,  self.no*3, 1)
    def forward(self, p3, p4, p5):
        return [self.head_p3(p3), self.head_p4(p4), self.head_p5(p5)]


# ── Modèle complet ────────────────────────────────────────────

class CustomYOLO(nn.Module):
    def __init__(self, num_classes=8, base_channels=128, pretrained=True):
        super().__init__()
        print("Construction YOLO avec backbone pré-entraîné...")
        self.backbone    = PretrainedBackbone(pretrained=pretrained)
        self.neck        = YOLOFPN(base_channels=base_channels)
        self.head        = DetectionHead(num_classes=num_classes, base_channels=base_channels)
        self.num_classes = num_classes
        self._init_neck_head()
        print("  ✅ Backbone ResNet50");  print("  ✅ Neck FPN/PAN");  print("  ✅ Head Detection")

    def _init_neck_head(self):
        for module in [self.neck, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1);  nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        print("Backbone gelé — entraînement head seulement")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True
        print("Backbone dégelé — entraînement complet")

    def forward(self, x):
        p3, p4, p5          = self.backbone(x)
        fpn_p3, pan_p4, pan_p5 = self.neck(p3, p4, p5)
        return self.head(fpn_p3, pan_p4, pan_p5)

    def summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*60}\n  ARCHITECTURE : {total:,} params ({total*4/1024/1024:.1f} MB)\n{'='*60}")


# ── Loss ──────────────────────────────────────────────────────

def bbox_iou_wh(wh1, wh2):
    inter = torch.min(wh1[0], wh2[0]) * torch.min(wh1[1], wh2[1])
    return inter / (wh1[0]*wh1[1] + wh2[0]*wh2[1] - inter + 1e-6)


class YOLOLossV2(nn.Module):
    def __init__(self, num_classes, img_size=640,
                 lambda_box=LAMBDA_BOX, lambda_obj=LAMBDA_OBJ,
                 lambda_cls=LAMBDA_CLS, lambda_noobj=LAMBDA_NOOBJ):
        super().__init__()
        self.num_classes  = num_classes;  self.img_size = img_size
        self.lambda_box   = lambda_box;   self.lambda_obj   = lambda_obj
        self.lambda_cls   = lambda_cls;   self.lambda_noobj = lambda_noobj
        self.anchors      = ANCHORS_NORM
        self.bce          = nn.BCEWithLogitsLoss(reduction='none')

    def focal_bce(self, pred, target, gamma=2.0):
        bce = self.bce(pred, target)
        p_t = target * torch.sigmoid(pred) + (1-target) * (1-torch.sigmoid(pred))
        return bce * ((1 - p_t) ** gamma)

    def forward(self, predictions, targets):
        device       = predictions[0].device
        scale_losses = []

        for scale_idx, pred in enumerate(predictions):
            B, C, H, W = pred.shape
            anchors    = self.anchors[scale_idx]
            anchor_t   = torch.tensor(anchors, dtype=torch.float32, device=device)
            pred       = pred.view(B, 3, 5+self.num_classes, H, W).permute(0,1,3,4,2).contiguous()

            obj_mask   = torch.zeros(B, 3, H, W, device=device, dtype=torch.bool)
            noobj_mask = torch.ones (B, 3, H, W, device=device, dtype=torch.bool)
            tx = torch.zeros(B,3,H,W,device=device);  ty = torch.zeros_like(tx)
            tw = torch.zeros_like(tx);                 th = torch.zeros_like(tx)
            tcls = torch.zeros(B, 3, H, W, self.num_classes, device=device)

            for b in range(B):
                if len(targets[b]) == 0: continue
                for t in targets[b].to(device):
                    cls_id = int(t[0]);  xc,yc,w,h = t[1].item(),t[2].item(),t[3].item(),t[4].item()
                    if w <= 0 or h <= 0: continue
                    wh_t   = torch.tensor([w,h], device=device)
                    best_a = int(torch.stack([bbox_iou_wh(wh_t, anchor_t[a]) for a in range(3)]).argmax())
                    gi     = max(0, min(W-1, int(xc*W)));  gj = max(0, min(H-1, int(yc*H)))
                    obj_mask[b,best_a,gj,gi]=True;  noobj_mask[b,best_a,gj,gi]=False
                    tx[b,best_a,gj,gi]=xc*W-gi;  ty[b,best_a,gj,gi]=yc*H-gj
                    aw,ah = anchors[best_a]
                    tw[b,best_a,gj,gi]=torch.log(torch.tensor(w/aw+1e-6,device=device))
                    th[b,best_a,gj,gi]=torch.log(torch.tensor(h/ah+1e-6,device=device))
                    tcls[b,best_a,gj,gi,cls_id]=1.0

            if obj_mask.sum() > 0:
                pred_xy   = torch.sigmoid(pred[...,0:2])
                loss_xy   = F.mse_loss(pred_xy[obj_mask], torch.stack([tx,ty],dim=-1)[obj_mask])
                loss_wh   = F.mse_loss(pred[...,2:4][obj_mask], torch.stack([tw,th],dim=-1)[obj_mask])
                loss_box  = loss_xy + loss_wh
                loss_cls  = self.focal_bce(pred[...,5:][obj_mask], tcls[obj_mask]).mean()
            else:
                loss_box = loss_cls = torch.tensor(0.0, device=device)

            pred_obj     = pred[...,4]
            loss_obj_pos = self.focal_bce(pred_obj[obj_mask],   obj_mask.float()[obj_mask]).mean()   if obj_mask.sum()>0   else torch.tensor(0.0,device=device)
            loss_obj_neg = self.focal_bce(pred_obj[noobj_mask], noobj_mask.float()[noobj_mask]*0).mean() if noobj_mask.sum()>0 else torch.tensor(0.0,device=device)

            scale_losses.append(
                self.lambda_box*loss_box + self.lambda_obj*loss_obj_pos +
                self.lambda_noobj*loss_obj_neg + self.lambda_cls*loss_cls
            )

        return sum(scale_losses) / 3.0
