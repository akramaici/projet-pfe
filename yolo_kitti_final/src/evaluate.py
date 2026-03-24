# ══════════════════════════════════════════════════════════════
# src/evaluate.py — Évaluation complète (mAP, latence, throughput)
# ══════════════════════════════════════════════════════════════

import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import ANCHORS_NORM, CONF_THRESH, NMS_THRESH, OUTPUTS_DIR


# ══════════════════════════════════════════════════════════════
# 1. NMS + DECODE
# ══════════════════════════════════════════════════════════════

def nms(boxes, scores, iou_threshold=0.35):
    if len(boxes)==0: return []
    x1=boxes[:,0]-boxes[:,2]/2;  y1=boxes[:,1]-boxes[:,3]/2
    x2=boxes[:,0]+boxes[:,2]/2;  y2=boxes[:,1]+boxes[:,3]/2
    areas=(x2-x1)*(y2-y1);  order=scores.argsort()[::-1];  keep=[]
    while len(order):
        i=order[0];  keep.append(i)
        if len(order)==1: break
        inter=np.maximum(0,np.minimum(x2[i],x2[order[1:]])-np.maximum(x1[i],x1[order[1:]])) * \
              np.maximum(0,np.minimum(y2[i],y2[order[1:]])-np.maximum(y1[i],y1[order[1:]]))
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order=order[1:][iou<iou_threshold]
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
        bx=(torch.sigmoid(pred[...,0])+gx)/W;  by=(torch.sigmoid(pred[...,1])+gy)/H
        bw=torch.exp(pred[...,2].clamp(-4,4))*anch[:,0].view(1,na,1,1)
        bh=torch.exp(pred[...,3].clamp(-4,4))*anch[:,1].view(1,na,1,1)
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

    final=[]
    for i in range(B):
        if not all_dets[i]: final.append(np.zeros((0,6))); continue
        dets=np.concatenate(all_dets[i],axis=0);  f=[]
        for cls_id in range(num_classes):
            mask=dets[:,5]==cls_id
            if mask.sum()==0: continue
            b=dets[mask,:4];  s=dets[mask,4]
            for k in nms(b,s,nms_thresh):
                f.append([b[k,0],b[k,1],b[k,2],b[k,3],s[k],float(cls_id)])
        final.append(np.array(f) if f else np.zeros((0,6)))
    return final


# ══════════════════════════════════════════════════════════════
# 2. mAP avec torchmetrics
# ══════════════════════════════════════════════════════════════

def compute_map_torchmetrics(model, val_loader, device, num_classes, class_names,
                              conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH, img_size=640):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError:
        import subprocess; subprocess.run(["pip","install","torchmetrics","-q"])
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

    print(f"\n{'='*60}\n  CALCUL mAP (torchmetrics)\n{'='*60}")

    metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        box_format="xywh", class_metrics=True
    ).to(device)
    model.eval()

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="mAP"):
            images = images.to(device)
            preds  = decode_predictions(model(images), conf_thresh, num_classes, nms_thresh, img_size)

            preds_fmt = []
            for p in preds:
                if len(p)==0:
                    preds_fmt.append({"boxes":torch.zeros((0,4),device=device),
                                      "scores":torch.zeros(0,device=device),
                                      "labels":torch.zeros(0,dtype=torch.int64,device=device)})
                else:
                    pt=torch.tensor(p,dtype=torch.float32,device=device)
                    b=pt[:,:4].clone()
                    b[:,0]-=b[:,2]/2;  b[:,1]-=b[:,3]/2
                    b[:,0]*=img_size;  b[:,1]*=img_size;  b[:,2]*=img_size;  b[:,3]*=img_size
                    preds_fmt.append({"boxes":b,"scores":pt[:,4],"labels":pt[:,5].long()})

            targets_fmt = []
            for t in targets:
                gt=t.to(device)
                if len(gt)==0:
                    targets_fmt.append({"boxes":torch.zeros((0,4),device=device),
                                        "labels":torch.zeros(0,dtype=torch.int64,device=device)})
                else:
                    b=gt[:,1:5].clone()
                    b[:,0]-=b[:,2]/2;  b[:,1]-=b[:,3]/2
                    b[:,0]*=img_size;  b[:,1]*=img_size;  b[:,2]*=img_size;  b[:,3]*=img_size
                    targets_fmt.append({"boxes":b,"labels":gt[:,0].long()})

            metric.update(preds_fmt, targets_fmt)

    results = metric.compute()
    map50   = results["map_50"].item()
    map5095 = results["map"].item()
    recall  = results.get("mar_100", torch.tensor(0)).item()

    print(f"\n🎯 mAP@50       = {map50:.4f}")
    print(f"🎯 mAP@50:95    = {map5095:.4f}")
    print(f"📊 Recall@100   = {recall:.4f}")

    print("\n📊 Toutes les métriques :")
    for k,v in results.items():
        if isinstance(v,torch.Tensor) and v.numel()==1:
            print(f"   {k:30s} : {v.item():.4f}")

    if "map_per_class" in results:
        print("\n📊 AP@50 par classe :")
        for cls_id in range(num_classes):
            name = class_names.get(cls_id,str(cls_id)) if isinstance(class_names,dict) else class_names[cls_id]
            ap50 = results["map_per_class"][cls_id].item() if cls_id<len(results["map_per_class"]) else 0
            print(f"   {name:20s} : AP@50 = {ap50:.4f}")

    if "mar_100_per_class" in results:
        print("\n📊 Recall@100 par classe :")
        for cls_id in range(num_classes):
            name = class_names.get(cls_id,str(cls_id)) if isinstance(class_names,dict) else class_names[cls_id]
            rec  = results["mar_100_per_class"][cls_id].item() if cls_id<len(results["mar_100_per_class"]) else 0
            print(f"   {name:20s} : Recall = {rec:.4f}")

    return map50, map5095, results


# ══════════════════════════════════════════════════════════════
# 3. Latence
# ══════════════════════════════════════════════════════════════

def measure_latency(model, device, img_size=640, n_warmup=5, n_runs=100):
    print(f"\n{'='*60}\n  MESURE DE LATENCE (1 image)\n{'='*60}")
    model.eval()
    dummy = torch.randn(1,3,img_size,img_size).to(device)

    print(f"🔥 Warmup ({n_warmup} passes)...")
    with torch.no_grad():
        for _ in range(n_warmup): model(dummy)
    if device=="cuda": torch.cuda.synchronize()

    latencies=[]
    print(f"⏱️  Mesure ({n_runs} passes)...")
    with torch.no_grad():
        for _ in range(n_runs):
            if device=="cuda": torch.cuda.synchronize()
            t0=time.perf_counter();  model(dummy)
            if device=="cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter()-t0)*1000)

    lat = np.array(latencies)
    res = {"min_ms":float(np.min(lat)),"max_ms":float(np.max(lat)),
           "mean_ms":float(np.mean(lat)),"median_ms":float(np.median(lat)),"std_ms":float(np.std(lat))}

    print(f"\n📊 Latence ({n_runs} mesures) :")
    print(f"   Min    : {res['min_ms']:.2f} ms")
    print(f"   Max    : {res['max_ms']:.2f} ms")
    print(f"   Mean   : {res['mean_ms']:.2f} ms")
    print(f"   Median : {res['median_ms']:.2f} ms")
    print(f"   Std    : {res['std_ms']:.2f} ms")
    print(f"   FPS    : {1000/res['mean_ms']:.1f} img/s")
    return res


# ══════════════════════════════════════════════════════════════
# 4. Throughput
# ══════════════════════════════════════════════════════════════

def measure_throughput(model, device, img_size=640,
                       batch_sizes=[1,2,4,8,16,32,64], n_warmup=5, n_runs=20):
    print(f"\n{'='*60}\n  MESURE DE THROUGHPUT\n{'='*60}")
    model.eval();  results=[]

    for bs in batch_sizes:
        print(f"\n  Batch size = {bs}")
        try:
            dummy=torch.randn(bs,3,img_size,img_size).to(device)
            with torch.no_grad():
                for _ in range(n_warmup): model(dummy)
            if device=="cuda": torch.cuda.synchronize()

            times=[]
            with torch.no_grad():
                for _ in range(n_runs):
                    if device=="cuda": torch.cuda.synchronize()
                    t0=time.perf_counter();  model(dummy)
                    if device=="cuda": torch.cuda.synchronize()
                    times.append(time.perf_counter()-t0)

            times=np.array(times);  tp=bs/times
            r={"batch_size":bs,"min_img_sec":float(np.min(tp)),"max_img_sec":float(np.max(tp)),
               "mean_img_sec":float(np.mean(tp)),"median_img_sec":float(np.median(tp)),
               "mean_lat_ms":float(np.mean(times)*1000)}
            results.append(r)
            print(f"    Min:{r['min_img_sec']:.1f} | Max:{r['max_img_sec']:.1f} | Mean:{r['mean_img_sec']:.1f} | Median:{r['median_img_sec']:.1f} img/s")
            del dummy
            if device=="cuda": torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    ❌ OOM — batch {bs}");
                if device=="cuda": torch.cuda.empty_cache();  break
            else: raise e

    return results


# ══════════════════════════════════════════════════════════════
# 5. Matrice de confusion
# ══════════════════════════════════════════════════════════════

def box_iou(b1, b2):
    x1=b1[0]-b1[2]/2;  y1=b1[1]-b1[3]/2;  x2=b1[0]+b1[2]/2;  y2=b1[1]+b1[3]/2
    x1g=b2[0]-b2[2]/2; y1g=b2[1]-b2[3]/2; x2g=b2[0]+b2[2]/2; y2g=b2[1]+b2[3]/2
    inter=max(0,min(x2,x2g)-max(x1,x1g))*max(0,min(y2,y2g)-max(y1,y1g))
    return inter/((x2-x1)*(y2-y1)+(x2g-x1g)*(y2g-y1g)-inter+1e-6)


def compute_confusion_matrix(model, val_loader, device, num_classes, class_names,
                              conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH,
                              iou_thresh=0.5, img_size=640, save_dir=None):
    save_dir = Path(save_dir or (OUTPUTS_DIR/"confusion"))
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    matrix = np.zeros((num_classes+1, num_classes+1), dtype=np.int32)

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Confusion Matrix"):
            images=images.to(device)
            preds=decode_predictions(model(images),conf_thresh,num_classes,nms_thresh,img_size)
            for pred,target in zip(preds,targets):
                gt=target.cpu().numpy()
                gt_boxes=gt[:,1:] if len(gt) else np.zeros((0,4))
                gt_cls=gt[:,0].astype(int) if len(gt) else np.array([],dtype=int)
                if len(pred)==0:
                    for c in gt_cls: matrix[c,num_classes]+=1; continue
                pred_boxes=pred[:,:4];  pred_cls=pred[:,5].astype(int)
                pred_conf=pred[:,4];    order=pred_conf.argsort()[::-1]
                pred_boxes=pred_boxes[order];  pred_cls=pred_cls[order]
                matched=set()
                for pb,pc in zip(pred_boxes,pred_cls):
                    best_iou=0;  best_gi=-1
                    for gi,(gb,gc) in enumerate(zip(gt_boxes,gt_cls)):
                        if gi in matched: continue
                        iou=box_iou(pb,gb)
                        if iou>best_iou: best_iou=iou;  best_gi=gi
                    if best_iou>=iou_thresh and best_gi>=0:
                        matrix[gt_cls[best_gi],pc]+=1;  matched.add(best_gi)
                    else:
                        matrix[num_classes,pc]+=1
                for gi,gc in enumerate(gt_cls):
                    if gi not in matched: matrix[gc,num_classes]+=1

    labels = [class_names.get(i,str(i)) if isinstance(class_names,dict) else class_names[i]
              for i in range(num_classes)] + ["background"]
    matrix_norm = matrix.astype(float)
    rs = matrix_norm.sum(axis=1,keepdims=True);  rs[rs==0]=1
    matrix_norm /= rs

    fig,axes=plt.subplots(1,2,figsize=(20,8))
    sns.heatmap(matrix,annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,ax=axes[0])
    axes[0].set_title("Matrice de confusion (valeurs brutes)")
    axes[0].set_xlabel("Prédit");  axes[0].set_ylabel("Réel")
    axes[0].tick_params(axis='x',rotation=45)
    sns.heatmap(matrix_norm,annot=True,fmt='.2f',cmap='Blues',xticklabels=labels,yticklabels=labels,
                vmin=0,vmax=1,ax=axes[1])
    axes[1].set_title("Matrice de confusion (normalisée)")
    axes[1].set_xlabel("Prédit");  axes[1].set_ylabel("Réel")
    axes[1].tick_params(axis='x',rotation=45)
    plt.suptitle("Matrice de confusion — YOLO KITTI",fontsize=15,fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir/"confusion_matrix.png",dpi=150,bbox_inches='tight');  plt.close()
    print(f"✅ Matrice de confusion : {save_dir/'confusion_matrix.png'}")

    print(f"\n{'Classe':20s} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Précision':>10} | {'Rappel':>8}")
    print("-"*70)
    for i in range(num_classes):
        tp=matrix[i,i];  fp=matrix[num_classes,i];  fn=matrix[i,num_classes]
        pr=tp/(tp+fp+1e-6);  rc=tp/(tp+fn+1e-6)
        name=labels[i]
        print(f"{name:20s} | {tp:>6} | {fp:>6} | {fn:>6} | {pr:>10.3f} | {rc:>8.3f}")

    return matrix


# ══════════════════════════════════════════════════════════════
# 6. Visualisation prédictions
# ══════════════════════════════════════════════════════════════

COLORS = {0:"white",1:"yellow",2:"cyan",3:"orange",4:"magenta",5:"red",6:"blue",7:"lime"}

def compute_iou_viz(b1,b2):
    x1m=b1[0]-b1[2]/2;  y1m=b1[1]-b1[3]/2;  x1M=b1[0]+b1[2]/2;  y1M=b1[1]+b1[3]/2
    x2m=b2[0]-b2[2]/2;  y2m=b2[1]-b2[3]/2;  x2M=b2[0]+b2[2]/2;  y2M=b2[1]+b2[3]/2
    inter=max(0,min(x1M,x2M)-max(x1m,x2m))*max(0,min(y1M,y2M)-max(y1m,y2m))
    return inter/(b1[2]*b1[3]+b2[2]*b2[3]-inter+1e-6)

def visualize_predictions(model, val_dataset, device, class_names, num_classes,
                          conf_threshold=CONF_THRESH, num_images=8, img_size=640,
                          save_dir=None):
    save_dir=Path(save_dir or (OUTPUTS_DIR/"predictions"))
    save_dir.mkdir(parents=True,exist_ok=True)
    model.eval()
    indices=np.random.choice(len(val_dataset),min(num_images,len(val_dataset)),replace=False)
    cols=4;  rows=(len(indices)+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(cols*5,rows*4))
    axes=axes.flatten() if rows>1 else [axes]

    for pi,ii in enumerate(indices):
        img_t,labels=val_dataset[ii]
        img_np=(img_t.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        with torch.no_grad(): raw=model(img_t.unsqueeze(0).to(device))
        pred_boxes=decode_predictions(raw,conf_thresh=0.4,num_classes=num_classes,img_size=img_size)[0]
        pred_boxes=sorted(pred_boxes,key=lambda x:-x[4])
        filtered=[]
        for box in pred_boxes:
            px,py,pw,ph,conf,_=box
            if conf<conf_threshold or pw*ph<0.003: continue
            if py<0.2 and pw*ph<0.015: continue
            if all(compute_iou_viz(box[:4],fb[:4])<=0.5 for fb in filtered):
                filtered.append(box)
        H,W=img_np.shape[:2];  ax=axes[pi];  ax.imshow(img_np)
        gt=labels.cpu().numpy() if isinstance(labels,torch.Tensor) else np.array(labels)
        for g in gt:
            _,cx,cy,bw,bh=g
            ax.add_patch(patches.Rectangle((int((cx-bw/2)*W),int((cy-bh/2)*H)),
                         int(bw*W),int(bh*H),linewidth=2,edgecolor="lime",facecolor="none"))
        for box in filtered:
            px,py,pw,ph,conf,pcls=box;  pcls=int(pcls)
            x1=int((px-pw/2)*W);  y1=int((py-ph/2)*H)
            color=COLORS.get(pcls,"white")
            ax.add_patch(patches.Rectangle((x1,y1),int(pw*W),int(ph*H),linewidth=2,edgecolor=color,facecolor="none"))
            name=class_names.get(pcls,str(pcls)) if isinstance(class_names,dict) else class_names[pcls]
            ax.text(x1,max(y1-4,10),f"{name} {conf:.2f}",color=color,fontsize=8,fontweight="bold",
                    bbox=dict(facecolor="black",alpha=0.4,pad=1,edgecolor="none"))
        ax.set_title(f"Img {ii} | {len(filtered)} dets",fontsize=9);  ax.axis("off")

    for i in range(len(indices),len(axes)): axes[i].axis("off")
    plt.suptitle("GT (vert) | Prédictions (couleur par classe)")
    plt.tight_layout()
    sp=save_dir/"predictions.png"
    plt.savefig(sp,dpi=150,bbox_inches="tight");  plt.close()
    print(f"✅ Visualisation : {sp}")


# ══════════════════════════════════════════════════════════════
# 7. Graphiques throughput
# ══════════════════════════════════════════════════════════════

def plot_throughput(throughput_results, save_dir=None):
    save_dir=Path(save_dir or (OUTPUTS_DIR/"performance"))
    save_dir.mkdir(parents=True,exist_ok=True)
    bs   =[r["batch_size"]    for r in throughput_results]
    means=[r["mean_img_sec"]  for r in throughput_results]
    mins =[r["min_img_sec"]   for r in throughput_results]
    maxs =[r["max_img_sec"]   for r in throughput_results]
    lpi  =[r["mean_lat_ms"]/r["batch_size"] for r in throughput_results]

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(bs,means,'b-o',linewidth=2,label='Mean')
    axes[0].fill_between(bs,mins,maxs,alpha=0.2,color='blue',label='Min-Max')
    axes[0].set_xlabel("Batch Size");  axes[0].set_ylabel("Throughput (img/s)")
    axes[0].set_title("Throughput vs Batch Size");  axes[0].legend();  axes[0].grid(alpha=0.3)
    axes[1].plot(bs,lpi,'r-o',linewidth=2)
    axes[1].set_xlabel("Batch Size");  axes[1].set_ylabel("Latence/image (ms)")
    axes[1].set_title("Latence par image vs Batch Size");  axes[1].grid(alpha=0.3)
    plt.suptitle("Performance système — YOLO KITTI",fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir/"throughput.png",dpi=150);  plt.close()
    print(f"✅ Graphique throughput : {save_dir/'throughput.png'}")


# ══════════════════════════════════════════════════════════════
# 8. Rapport complet
# ══════════════════════════════════════════════════════════════

def print_full_report(map50, map5095, latency_results, throughput_results):
    print(f"\n{'█'*60}\n  RAPPORT D'ÉVALUATION COMPLET\n{'█'*60}")
    print(f"\n📊 1. MÉTRIQUES DE PRÉCISION\n{'-'*40}")
    print(f"   mAP@50    : {map50:.4f}  ({map50*100:.2f}%)")
    print(f"   mAP@50:95 : {map5095:.4f}  ({map5095*100:.2f}%)")
    print(f"\n⏱️  2. LATENCE (1 image)\n{'-'*40}")
    print(f"   Min    : {latency_results['min_ms']:.2f} ms")
    print(f"   Max    : {latency_results['max_ms']:.2f} ms")
    print(f"   Mean   : {latency_results['mean_ms']:.2f} ms")
    print(f"   Median : {latency_results['median_ms']:.2f} ms")
    print(f"   FPS    : {1000/latency_results['mean_ms']:.1f} img/s")
    print(f"\n🚀 3. THROUGHPUT (par batch size)\n{'-'*40}")
    df=pd.DataFrame(throughput_results)
    df.columns=["Batch","Min(img/s)","Max(img/s)","Mean(img/s)","Median(img/s)","Lat(ms)"]
    print(df.to_string(index=False))
    best=max(throughput_results,key=lambda x:x["mean_img_sec"])
    print(f"\n   ✅ Meilleur batch size : {best['batch_size']} ({best['mean_img_sec']:.1f} img/s)")
    print(f"\n{'█'*60}")


def run_evaluation(model, val_loader, val_dataset, class_names, num_classes,
                   device, img_size=640, save_dir=None):
    save_dir = Path(save_dir or OUTPUTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # mAP
    map50, map5095, _ = compute_map_torchmetrics(
        model, val_loader, device, num_classes, class_names, img_size=img_size)

    # Latence
    latency = measure_latency(model, device, img_size)

    # Throughput
    throughput = measure_throughput(model, device, img_size)

    # Graphiques
    plot_throughput(throughput, save_dir=save_dir/"performance")

    # Rapport
    print_full_report(map50, map5095, latency, throughput)

    # Matrice de confusion
    compute_confusion_matrix(model, val_loader, device, num_classes, class_names,
                             img_size=img_size, save_dir=save_dir/"confusion")

    # Visualisation
    visualize_predictions(model, val_dataset, device, class_names, num_classes,
                          img_size=img_size, save_dir=save_dir/"predictions")

    return map50, map5095
