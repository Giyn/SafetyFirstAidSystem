from control import Control
from models.experimental import *
from utils.datasets import *
from utils.utils import *


def detect(save_img=False):
    source = '0'
    out = 'inference/output'
    weights = './weights/best.pt'

    save_txt = False
    imgsz = 160
    webcam = True
    conf_thres = 0.4
    iou_thres = 0.5

    con = Control("COM3")

    # Initialize
    device = torch.device('cpu')
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder
    # half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    vid_path, vid_writer = None, None

    view_img = True
    dataset = LoadStreams(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in
              range(len(names))]
    half = False
    # Run inference
    t0 = time.time()
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    n = 0
    for path, img, im0s, vid_cap in dataset:
        n = n + 1
        if n % 2 == 0:
            continue
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,
                                   agnostic=False)
        t2 = torch_utils.time_synchronized()

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + (
                '_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[
                [1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                size = 0
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(
                            torch.tensor(xyxy).view(1, 4)) / gn).view(
                            -1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (
                            cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)], line_thickness=3)
                        width = int(xyxy[2].cpu().detach().numpy()) - int(
                            xyxy[0].cpu().detach().numpy())
                        height = int(xyxy[3].cpu().detach().numpy()) - int(
                            xyxy[1].cpu().detach().numpy())

                        s_temp = width * height

                        if s_temp > size:
                            size = s_temp
                            x = (int(xyxy[0].cpu().detach().numpy()) + int(
                                xyxy[2].cpu().detach().numpy())) // 2
                            y = (int(xyxy[1].cpu().detach().numpy()) + int(
                                xyxy[3].cpu().detach().numpy())) // 2

                            print(x, y, xyxy)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                try:
                    con.update(y, x)
                except:
                    pass

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    detect()
