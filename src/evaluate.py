# ══════════════════════════════════════════════════════════════
# src/evaluate.py — mAP50 + visualisation
# ══════════════════════════════════════════════════════════════

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import ANCHORS_NORM, CONF_THRESH, NMS_THRESH, OUTPUTS_DIR


def nms(boxes, scores, iou_threshold=0.35):
    if len(boxes) == 0: return []
    x1=boxes[:,0]-boxes[:,2]/2;  y1=boxes[:,1]-boxes[:,3]/2
    x2=boxes[:,0]+boxes[:,2]/2;  y2=boxes[:,1]+boxes[:,3]/2
    areas=( x2-x1)*(y2-y1);  order=scores.argsort()[::-1];  keep=[]
    while len(order):
        i=order[0];  keep.append(i)
        if len(order)==1: break
        inter = np.maximum(0,np.minimum(x2[i],x2[order[1:]])-np.maximum(x1[i],x1[order[1:]])) * \
                np.maximum(0,np.minimum(y2[i],y2[order[1:]])-np.maximum(y1[i],y1[order[1:]]))
        iou   = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order = order[1:][iou < iou_threshold]
    return keep


def decode_predictions(raw_output, conf_thresh=CONF_THRESH, num_classes=8,
                       nms_thresh=NMS_THRESH, img_size=640):
    scales   = raw_output if isinstance(raw_output,(list,tuple)) else [raw_output]
    B        = scales[0].shape[0]
    all_dets = [[] for _ in range(B)]

    for scale_pred, anchors_norm in zip(scales, ANCHORS_NORM):
        B2,_,H,W = scale_pred.shape;  na=3
        pred = scale_pred.view(B2,na,5+num_classes,H,W).permute(0,1,3,4,2)
        gy,gx = torch.meshgrid(torch.arange(H,device=pred.device),
                               torch.arange(W,device=pred.device), indexing="ij")
        anch  = torch.tensor(anchors_norm, dtype=torch.float32, device=pred.device)
        bx = (torch.sigmoid(pred[...,0])+gx)/W;  by = (torch.sigmoid(pred[...,1])+gy)/H
        bw = torch.exp(pred[...,2].clamp(-4,4))*anch[:,0].view(1,na,1,1)
        bh = torch.exp(pred[...,3].clamp(-4,4))*anch[:,1].view(1,na,1,1)
        obj=torch.sigmoid(pred[...,4]);  cls=torch.sigmoid(pred[...,5:])
        scores=obj*cls.max(dim=-1).values;  cls_ids=cls.max(dim=-1).indices
        for i in range(B2):
            mask=scores[i]>conf_thresh
            if mask.sum()==0: continue
            boxes=torch.stack([bx[i],by[i],bw[i],bh[i]],dim=-1)[mask].cpu().numpy()
            conf =scores[i][mask].cpu().numpy();  clsid=cls_ids[i][mask].cpu().numpy()
            dets =np.concatenate([boxes,conf[:,None],clsid[:,None]],axis=1)
            dets =dets[(dets[:,2]*dets[:,3])>0.001]
            if len(dets): all_dets[i].append(dets)

    final_preds=[]
    for i in range(B):
        if not all_dets[i]: final_preds.append(np.zeros((0,6))); continue
        dets=np.concatenate(all_dets[i],axis=0);  final=[]
        for cls_id in range(num_classes):
            mask=dets[:,5]==cls_id
            if mask.sum()==0: continue
            b=dets[mask,:4];  s=dets[mask,4]
            for k in nms(b,s,nms_thresh):
                final.append([b[k,0],b[k,1],b[k,2],b[k,3],s[k],float(cls_id)])
        final_preds.append(np.array(final) if final else np.zeros((0,6)))
    return final_preds


def box_iou(b1,b2):
    x1=b1[0]-b1[2]/2;  y1=b1[1]-b1[3]/2;  x2=b1[0]+b1[2]/2;  y2=b1[1]+b1[3]/2
    x1g=b2[0]-b2[2]/2; y1g=b2[1]-b2[3]/2; x2g=b2[0]+b2[2]/2; y2g=b2[1]+b2[3]/2
    inter=max(0,min(x2,x2g)-max(x1,x1g))*max(0,min(y2,y2g)-max(y1,y1g))
    return inter/((x2-x1)*(y2-y1)+(x2g-x1g)*(y2g-y1g)-inter+1e-6)


def compute_map(model, val_loader, device, num_classes, iou_thresh=0.5,
                conf_thresh=CONF_THRESH, img_size=640):
    model.eval()
    all_tp=defaultdict(list);  all_conf=defaultdict(list);  all_gt=defaultdict(int)
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="mAP"):
            images=images.to(device)
            preds=decode_predictions(model(images),conf_thresh=conf_thresh,
                                     num_classes=num_classes,img_size=img_size)
            for p,t in zip(preds,targets):
                gt=t.cpu().numpy()
                for cls_id in range(num_classes):
                    gt_cls=gt[gt[:,0]==cls_id][:,1:] if len(gt) else np.zeros((0,4))
                    all_gt[cls_id]+=len(gt_cls)
                    pred_cls=p[p[:,5]==cls_id] if len(p) else np.zeros((0,6))
                    if len(pred_cls):
                        for pred in pred_cls[pred_cls[:,4].argsort()[::-1]]:
                            best_iou=max([box_iou(pred[:4],g) for g in gt_cls],default=0)
                            all_tp[cls_id].append(int(best_iou>=iou_thresh))
                            all_conf[cls_id].append(float(pred[4]))
    aps={}
    for cls_id in range(num_classes):
        if all_gt[cls_id]==0: continue
        tp=np.array(all_tp[cls_id]);  conf=np.array(all_conf[cls_id])
        if len(tp)==0: aps[cls_id]=0; continue
        tp=tp[conf.argsort()[::-1]]
        cum_tp=np.cumsum(tp);  cum_fp=np.cumsum(1-tp)
        prec=cum_tp/(cum_tp+cum_fp+1e-6);  rec=cum_tp/(all_gt[cls_id]+1e-6)
        aps[cls_id]=float(np.trapezoid(prec,rec))
    return np.mean(list(aps.values())) if aps else 0, aps


COLORS={0:"white",1:"yellow",2:"cyan",3:"orange",4:"magenta",5:"red",6:"blue",7:"lime"}

def visualize_predictions(model, val_dataset, device, class_names, num_classes,
                          conf_threshold=CONF_THRESH, num_images=8, img_size=640,
                          save_dir=None):
    save_dir = Path(save_dir or (OUTPUTS_DIR / "predictions"))
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    indices = np.random.choice(len(val_dataset), min(num_images,len(val_dataset)), replace=False)
    cols=4;  rows=(len(indices)+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(cols*5,rows*4))
    axes=axes.flatten() if rows>1 else [axes]

    for plot_idx,img_idx in enumerate(indices):
        img_tensor,labels=val_dataset[img_idx]
        img_np=(img_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        with torch.no_grad():
            raw=model(img_tensor.unsqueeze(0).to(device))
        pred_boxes=decode_predictions(raw,conf_thresh=0.4,num_classes=num_classes,img_size=img_size)[0]
        pred_boxes=sorted(pred_boxes,key=lambda x:-x[4])
        filtered=[]
        for box in pred_boxes:
            px,py,pw,ph,conf,_=box
            if conf<conf_threshold or pw*ph<0.003: continue
            if py<0.2 and pw*ph<0.015: continue
            if all(box_iou(box[:4],fb[:4])<=0.5 for fb in filtered):
                filtered.append(box)
        H,W=img_np.shape[:2];  ax=axes[plot_idx];  ax.imshow(img_np)
        gt_arr=labels.cpu().numpy() if isinstance(labels,torch.Tensor) else np.array(labels)
        for g in gt_arr:
            _,cx,cy,bw,bh=g
            ax.add_patch(patches.Rectangle((int((cx-bw/2)*W),int((cy-bh/2)*H)),
                         int(bw*W),int(bh*H),linewidth=2,edgecolor="lime",facecolor="none"))
        for box in filtered:
            px,py,pw,ph,conf,pcls=box;  pcls=int(pcls)
            x1=int((px-pw/2)*W);  y1=int((py-ph/2)*H)
            color=COLORS.get(pcls,"white")
            ax.add_patch(patches.Rectangle((x1,y1),int(pw*W),int(ph*H),
                         linewidth=2,edgecolor=color,facecolor="none"))
            name=class_names.get(pcls,str(pcls)) if isinstance(class_names,dict) else class_names[pcls]
            ax.text(x1,max(y1-4,10),f"{name} {conf:.2f}",color=color,fontsize=8,fontweight="bold",
                    bbox=dict(facecolor="black",alpha=0.4,pad=1,edgecolor="none"))
        ax.set_title(f"Img {img_idx} | {len(filtered)} dets",fontsize=9);  ax.axis("off")

    for i in range(len(indices),len(axes)): axes[i].axis("off")
    plt.suptitle("GT (vert) | Prédictions (couleur par classe)")
    plt.tight_layout()
    save_path=save_dir/"predictions.png"
    plt.savefig(save_path,dpi=150,bbox_inches="tight");  plt.close()
    print(f"✅ Visualisation : {save_path}")


def run_evaluation(model, val_loader, val_dataset, class_names, num_classes,
                   device, img_size=640):
    print("\n" + "="*60);  print("  ÉVALUATION");  print("="*60)
    map50, aps = compute_map(model, val_loader, device, num_classes, img_size=img_size)
    print(f"\n🎯 mAP50 = {map50:.4f}")
    for cls_id, ap in aps.items():
        name=class_names.get(cls_id,str(cls_id)) if isinstance(class_names,dict) else class_names[cls_id]
        print(f"   {name:20s} : AP = {ap:.4f}")
    visualize_predictions(model, val_dataset, device, class_names, num_classes,
                          img_size=img_size)
    return map50, aps
