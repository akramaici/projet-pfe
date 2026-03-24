# ══════════════════════════════════════════════════════════════
# src/inference.py — Inférence image / vidéo
# ══════════════════════════════════════════════════════════════

import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import IMG_SIZE, CONF_THRESH, NMS_THRESH
from src.dataset  import letterbox
from src.evaluate import decode_predictions

COLORS_MPL = {0:"white",1:"yellow",2:"cyan",3:"orange",4:"magenta",5:"red",6:"blue",7:"lime"}
COLORS_CV2 = {0:(255,255,255),1:(0,255,255),2:(255,255,0),3:(0,165,255),
              4:(255,0,255),5:(0,0,255),6:(255,0,0),7:(0,255,0)}

def get_name(cn, cls_id):
    if isinstance(cn,dict): return cn.get(cls_id,str(cls_id))
    return cn[cls_id] if cls_id<len(cn) else str(cls_id)


def predict_image(model, image_path, device, class_names, num_classes,
                  img_size=IMG_SIZE, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH,
                  save_path="outputs/test_image.png"):
    img=cv2.imread(str(image_path))
    if img is None: raise FileNotFoundError(f"❌ {image_path}")
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);  H0,W0=img_rgb.shape[:2]
    print(f"📷 Image : {W0}x{H0}")
    img_lb,scale,px,py=letterbox(img_rgb,img_size)
    img_t=torch.from_numpy(img_lb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad(): raw=model(img_t)
    dets=decode_predictions(raw,conf_thresh,num_classes,nms_thresh,img_size)[0]
    print(f"🎯 {len(dets)} détections")
    fig,ax=plt.subplots(1,1,figsize=(14,6));  ax.imshow(img_rgb)
    for det in dets:
        bpx,bpy,pw,ph,conf,cls_id=det;  cls_id=int(cls_id)
        x1=int(((bpx-pw/2)*img_size-px)/scale);  y1=int(((bpy-ph/2)*img_size-py)/scale)
        x2=int(((bpx+pw/2)*img_size-px)/scale);  y2=int(((bpy+ph/2)*img_size-py)/scale)
        x1,y1=max(0,x1),max(0,y1);  x2,y2=min(W0,x2),min(H0,y2)
        color=COLORS_MPL.get(cls_id,"white");  name=get_name(class_names,cls_id)
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=3,edgecolor=color,facecolor="none"))
        ax.text(x1,max(y1-6,12),f"{name} {conf:.2f}",color=color,fontsize=11,fontweight="bold",
                bbox=dict(facecolor="black",alpha=0.5,pad=2,edgecolor="none"))
    ax.axis("off");  ax.set_title(f"{len(dets)} détections — conf > {conf_thresh}",fontsize=13)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(save_path,dpi=150);  plt.close()
    print(f"✅ Sauvegardé : {save_path}")


def predict_video(model, video_path, device, class_names, num_classes,
                  img_size=IMG_SIZE, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH,
                  output_path="outputs/output_video.mp4"):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise FileNotFoundError(f"❌ {video_path}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));  H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS=cap.get(cv2.CAP_PROP_FPS);  TOTAL=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    out=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))
    model.eval();  fi=0
    print(f"📹 {W}x{H} @ {FPS:.1f}fps — {TOTAL} frames")
    while True:
        ret,frame=cap.read()
        if not ret: break
        img_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_lb,scale,px,py=letterbox(img_rgb,img_size)
        img_t=torch.from_numpy(img_lb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad(): raw=model(img_t)
        dets=decode_predictions(raw,conf_thresh,num_classes,nms_thresh,img_size)[0]
        for det in dets:
            bpx,bpy,pw,ph,conf,cls_id=det;  cls_id=int(cls_id)
            x1=int(((bpx-pw/2)*img_size-px)/scale);  y1=int(((bpy-ph/2)*img_size-py)/scale)
            x2=int(((bpx+pw/2)*img_size-px)/scale);  y2=int(((bpy+ph/2)*img_size-py)/scale)
            x1,y1=max(0,x1),max(0,y1);  x2,y2=min(W,x2),min(H,y2)
            color=COLORS_CV2.get(cls_id,(255,255,255));  name=get_name(class_names,cls_id)
            label=f"{name} {conf:.2f}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            cv2.rectangle(frame,(x1,y1-th-6),(x1+tw,y1),color,-1)
            cv2.putText(frame,label,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        out.write(frame);  fi+=1
        if fi%50==0: print(f"  {fi}/{TOTAL} frames...")
    cap.release();  out.release()
    print(f"✅ Vidéo : {output_path}")
