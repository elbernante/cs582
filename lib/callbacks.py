import os, pickle
import cv2
import numpy as np
from keras.callbacks import Callback

from .file_io import create_if_not_exists

class TrainMonitor(Callback):
    """Callback that records events into a `History` object.
    Saves a sample prediction at the end of each epoch.
    """

    def __init__(self, log_file, 
                 x_sample=None, y_sample=None, 
                 out_dir="output/preds/"):
        super(TrainMonitor, self).__init__()
        self.log_file = log_file
        self.x_sample = x_sample
        self.y_sample = y_sample
        self._preped_img = None
        self.out_dir = out_dir

        create_if_not_exists(out_dir)

        if x_sample is not None:
            self._preped_img =  self._get_img_0(x_sample.squeeze(),
                                                y_sample.squeeze())

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self._save_history()
        self._generate_out_img();

    def _save_history(self):
        with open(self.log_file, 'wb') as f:
            pickle.dump({**self.history, 'epoch': self.epoch}, f)

    def _generate_out_img(self):
        if self.x_sample is not None:
            pred = self.model.predict(self.x_sample, batch_size=1)
            self._save_sample_pred(pred.squeeze())

    def _save_sample_pred(self, soft_pred):
        src_img = self._preped_img

        jet = lambda i: cv2.applyColorMap(i, cv2.COLORMAP_JET)
        hot = lambda i: cv2.applyColorMap(i, cv2.COLORMAP_HOT)
        gray = lambda i: cv2.cvtColor((i * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        soft = lambda i: (((i - min(0, i.min())) / (max(1, i.max()) - min(0, i.min()))) * 255).astype(np.uint8)
        scale = lambda i: (((i - i.min()) / (i.max() - i.min())) * 255).astype(np.uint8)

        pred = (soft_pred >= 0.5).astype(np.uint8)  # threshold
        
        diff = np.zeros_like(pred, dtype=np.uint8) if src_img is None \
                    else ((src_img[2] * 2.) + pred.astype(np.float32)) / 3.0

        monitor = np.concatenate((jet(soft(soft_pred)), jet(scale(soft_pred)), hot(soft(diff))), axis=1)
        pred_img = gray(pred)
        text_area = np.concatenate((np.zeros_like(pred_img, dtype=np.uint8) + 63,
                                    np.zeros_like(pred_img, dtype=np.uint8) + 63,
                                    pred_img), axis=1)
        cv2.putText(text_area, "Prediction at epoch: {}".format(self.epoch[-1] + 1), 
                    (10, text_area.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (253, 253, 253), 1, cv2.LINE_AA)
        
        if src_img is not None:
            h = ((src_img[1] * 0.9) + 0.1) * pred.astype(np.float32)
            d = src_img[1] * 0.4 * np.logical_not(pred).astype(np.float32)
            overlay = d + h
        
            src = np.concatenate([src_img[0][0], src_img[0][1], gray(overlay)], axis=1)
            imgs = np.concatenate((src, monitor, text_area), axis=0)
        else:
            imgs = np.concatenate((monitor, text_area), axis=0)
        
        fname = '{}.tif'.format(self.epoch[-1] + 1)
        cv2.imwrite(os.path.join(self.out_dir, fname), imgs)

    def _get_img_0(self, img, lbl):
        img = (img - img.min())/(img.max() - img.min())
       
        h = ((img * 0.9) + 0.1) * lbl.astype(np.float32)
        d = img * 0.4 * np.logical_not(lbl).astype(np.float32)
        overlay = d + h

        cvt = lambda i: cv2.cvtColor((i * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return [cvt(i) for i in [img, overlay]], img, lbl.astype(np.float32)
