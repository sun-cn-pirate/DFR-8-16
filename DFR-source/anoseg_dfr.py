import torch
import torch.nn as nn
import torch.nn.functional as F
#from extractors.feature import Extractor
from feature import Extractor
from torch.utils.data import DataLoader
import torch.optim as optim
#from data.MVTec import NormalDataset, TrainTestDataset

import time
import datetime
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage import measure
from skimage.transform import resize
from skimage.util import img_as_ubyte
import pandas as pd

from feat_cae import FeatCAE

import joblib
from sklearn.decomposition import PCA

from utils import *


class AnoSegDFR():
    """
    Anomaly segmentation model: DFR.
    """
    def __init__(self, cfg):
        super(AnoSegDFR, self).__init__()
        self.cfg = cfg
        self.path = cfg.save_path    # model and results saving path

        self.n_layers = len(cfg.cnn_layers)
        self.data_name = cfg.data_name
        self.n_dim = cfg.latent_dim
        if self.n_dim is None and getattr(cfg, 'resume', False):
            self.n_dim = self.find_saved_dimension()

        self.log_step = 10

        self.img_size = cfg.img_size
        self.threshold = cfg.thred
        self.device = torch.device(cfg.device)
        self.training_elapsed_seconds = 0.0

        # feature extractor
        self.extractor = Extractor(backbone=cfg.backbone,
                 cnn_layers=cfg.cnn_layers,
                 upsample=cfg.upsample,
                 is_agg=cfg.is_agg,
                 kernel_size=cfg.kernel_size,
                 stride=cfg.stride,
                 dilation=cfg.dilation,
                 featmap_size=cfg.featmap_size,
                 device=cfg.device).to(self.device)

        # datasest
        self.train_data_path = cfg.train_data_path
        self.test_data_path = cfg.test_data_path
        self.train_data = self.build_dataset(is_train=True)
        self.test_data = self.build_dataset(is_train=False)

        # dataloader
        workers = getattr(cfg, "workers", 4)
        pin_memory = self.device.type == "cuda"
        self.train_data_loader = DataLoader(
            self.train_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )
        self.test_data_loader = DataLoader(
            self.test_data,
            batch_size=1,
            shuffle=False,
            num_workers=min(workers, 2),
            pin_memory=pin_memory,
        )
        self.eval_data_loader = DataLoader(
            self.train_data,
            batch_size=10,
            shuffle=False,
            num_workers=min(workers, 2),
            pin_memory=pin_memory,
        )


        # autoencoder classifier
        self.autoencoder, self.model_name = self.build_classifier()
        if cfg.model_name != "":
            self.model_name = cfg.model_name
        print("model name:", self.model_name)

        # optimizer
        self.lr = cfg.lr
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=0)

        # saving paths
        self.subpath = self.data_name + "/" + self.model_name
        self.model_path = os.path.join(self.path, "models/" + self.subpath + "/model")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.eval_path = os.path.join(self.path, "models/" + self.subpath + "/eval")
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)

    def build_classifier(self):
        # self.load_dim(self.model_path)
        if self.n_dim is None:
            print("Estimating one class classifier AE parameter...")
            feats = torch.Tensor()
            self.extractor.eval()
            with torch.no_grad():
                for i, normal_img in enumerate(self.eval_data_loader):
                    if i > 0:
                        break
                    normal_img = normal_img.to(self.device, non_blocking=True)
                    feat = self.extractor.feat_vec(normal_img)
                    feats = torch.cat([feats, feat.cpu()], dim=0)
            # to numpy
            feats = feats.detach().numpy()
            # estimate parameters for mlp
            pca = PCA(n_components=0.90)    # 0.9 here try 0.8
            pca.fit(feats)
            n_dim, in_feat = pca.components_.shape
            print("AE Parameter (in_feat, n_dim): ({}, {})".format(in_feat, n_dim))
            self.n_dim = n_dim
        else:
            self.extractor.eval()
            with torch.no_grad():
                for i, normal_img in enumerate(self.eval_data_loader):
                    if i > 0:
                        break
                    normal_img = normal_img.to(self.device, non_blocking=True)
                    feat = self.extractor.feat_vec(normal_img)
            in_feat = feat.shape[1]

        print("BN?:", self.cfg.is_bn)
        autoencoder = FeatCAE(in_channels=in_feat, latent_dim=self.n_dim, is_bn=self.cfg.is_bn).to(self.device)
        model_name = "AnoSegDFR({})_{}_l{}_d{}_s{}_k{}_{}".format('BN' if self.cfg.is_bn else 'noBN',
                                                                self.cfg.backbone, self.n_layers,
                                                                self.n_dim, self.cfg.stride[0],
                                                                self.cfg.kernel_size[0], self.cfg.upsample)

        return autoencoder, model_name

    def find_saved_dimension(self):
        category_path = Path(self.path) / "models" / self.data_name
        if self.cfg.model_name:
            candidates = [category_path / self.cfg.model_name / "model" / "n_dim.npy"]
        else:
            bn_name = "BN" if self.cfg.is_bn else "noBN"
            pattern = (
                f"AnoSegDFR({bn_name})_{self.cfg.backbone}_l{self.n_layers}_d*"
                f"_s{self.cfg.stride[0]}_k{self.cfg.kernel_size[0]}_"
                f"{self.cfg.upsample}/model/n_dim.npy"
            )
            candidates = sorted(category_path.glob(pattern))
        candidates = [path for path in candidates if path.is_file()]
        if not candidates:
            return None
        if len(candidates) > 1:
            rendered = ", ".join(str(path) for path in candidates)
            raise ValueError(
                "Multiple resumable latent dimensions match this configuration; "
                f"set --latent-dim or --model-name explicitly: {rendered}"
            )
        n_dim = int(np.load(candidates[0], allow_pickle=False).item())
        print(f"Using saved latent dimension {n_dim}: {candidates[0]}")
        return n_dim

    def build_dataset(self, is_train):
        from MVTec import NormalDataset, TestDataset
        normal_data_path = self.train_data_path
        abnormal_data_path = self.test_data_path
        if is_train:
            dataset = NormalDataset(normal_data_path, normalize=True)
        else:
            dataset = TestDataset(path=abnormal_data_path)
        return dataset

    def train(self, resume=True):
        start_time = time.time()
        elapsed_before_resume = self.training_elapsed_seconds
        iters_per_epoch = len(self.train_data_loader)  # total iterations every epoch
        epochs = self.cfg.epochs  # total epochs
        start_epoch = self.load_training_checkpoint() + 1 if resume else 1
        if start_epoch > epochs:
            print(f"Training already complete at epoch {start_epoch - 1}.")
            self.plot_loss_curve()
            return

        for epoch in range(start_epoch, epochs+1):
            self.extractor.eval()
            self.autoencoder.train()
            losses = []
            for i, normal_img in enumerate(self.train_data_loader):
                normal_img = normal_img.to(self.device, non_blocking=True)
                # forward and backward
                total_loss = self.optimize_step(normal_img)

                # statistics and logging
                loss = {}
                loss['total_loss'] = total_loss.item()
                
                # tracking loss
                losses.append(loss['total_loss'])
            
            if epoch % 5 == 0:
                #                 self.save_model()

                print('Epoch {}/{}'.format(epoch, epochs))
                print('-' * 10)
                elapsed = time.time() - start_time
                total_time = ((epochs * iters_per_epoch) - (epoch * iters_per_epoch + i)) * elapsed / (
                        epoch * iters_per_epoch + i + 1)
                epoch_time = (iters_per_epoch - i) * elapsed / (epoch * iters_per_epoch + i + 1)

                epoch_time = str(datetime.timedelta(seconds=epoch_time))
                total_time = str(datetime.timedelta(seconds=total_time))
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}]".format(
                    elapsed, epoch_time, total_time, epoch, epochs, i + 1, iters_per_epoch)

                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            checkpoint_every = getattr(self.cfg, "checkpoint_every", 10)
            if epoch % checkpoint_every == 0:
                # save model
                self.training_elapsed_seconds = (
                    elapsed_before_resume + time.time() - start_time
                )
                self.save_model(epoch)
                self.validation(epoch)

#             print("Cost total time {}s".format(time.time() - start_time))
#             print("Done.")
            self.tracking_loss(epoch, np.mean(np.array(losses)))

        # save model
        self.training_elapsed_seconds = (
            elapsed_before_resume + time.time() - start_time
        )
        self.save_model(epochs)
        self.plot_loss_curve()
        print("Cost total time {}s".format(time.time() - start_time))
        print("Done.")

    def tracking_loss(self, epoch, loss):
        out_file = os.path.join(self.eval_path, '{}_epoch_loss.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",loss" + "\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + str(loss) + "\n")

    def plot_loss_curve(self):
        loss_path = os.path.join(
            self.eval_path, '{}_epoch_loss.csv'.format(self.model_name)
        )
        if not os.path.isfile(loss_path):
            return
        history = pd.read_csv(loss_path)
        if history.empty:
            return
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(history['Epoch'], history['loss'], linewidth=1.25)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Reconstruction loss')
        axis.set_title(f'DFR training loss: {self.data_name}')
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(
            os.path.join(self.eval_path, f'{self.model_name}_loss.png'),
            dpi=160,
        )
        plt.close(figure)

    def optimize_step(self, input_data):
        self.extractor.eval()
        self.autoencoder.train()

        self.optimizer.zero_grad()

        # forward
        with torch.no_grad():
            input_data = self.extractor(input_data)

        # print(input_data.size())
        dec = self.autoencoder(input_data)

        # loss
        total_loss = self.autoencoder.loss_function(dec, input_data)

        # self.reset_grad()
        total_loss.backward()

        self.optimizer.step()

        return total_loss

    def score(self, input):
        """
        Args:
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map with shape (img_size_h, img_size_w)
        """
        self.extractor.eval()
        self.autoencoder.eval()

        with torch.no_grad():
            input = self.extractor(input)
            dec = self.autoencoder(input)
            scores = self.autoencoder.compute_energy(dec, input)
        scores = scores.reshape((1, 1, self.extractor.out_size[0], self.extractor.out_size[1]))    # test batch size is 1.
        scores = nn.functional.interpolate(scores, size=self.img_size, mode="bilinear", align_corners=True).squeeze()
        # print("score shape:", scores.shape)
        return scores

    def segment(self, input, threshold=0.5):
        """
        Args:
            input: image with size of (img_size_h, img_size_w, channels)
        Returns:
            score map and binary score map with shape (img_size_h, img_size_w)
        """
        # predict
        scores = self.score(input).detach().cpu().numpy()

        # binary score
        print("threshold:", threshold)
        binary_scores = np.zeros_like(scores)    # torch.zeros_like(scores)
        binary_scores[scores <= threshold] = 0
        binary_scores[scores > threshold] = 1

        return scores, binary_scores

    def segment_evaluation(self):
        i = 0
        metrics = []
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=self.threshold)

            # show something
            #     plt.figure()
            #     ax1 = plt.subplot(1, 2, 1)
            #     ax1.imshow(resize(mask[0], (256, 256)))
            #     ax1.set_title("gt")

            #     ax2 = plt.subplot(1, 2, 2)
            #     ax2.imshow(scores)
            #     ax2.set_title("pred")

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_seg_results(normalize(scores), binary_scores, mask, name)
            # metrics of one batch
            if name.split("/")[-2] != "good":
                specificity, sensitivity, accuracy, coverage, auc = spec_sensi_acc_iou_auc(mask, binary_scores, scores)
                metrics.append([specificity, sensitivity, accuracy, coverage, auc])
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        # metrics over all data
        metrics = np.array(metrics)
        metrics_mean = metrics.mean(axis=0)
        metrics_std = metrics.std(axis=0)
        print("metrics: specificity, sensitivity, accuracy, iou, auc")
        print("mean:", metrics_mean)
        print("std:", metrics_std)
        print("threshold:", self.threshold)

    def save_paths(self):
        # generating saving paths
        score_map_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/score_map")
        if not os.path.exists(score_map_path):
            os.makedirs(score_map_path)

        binary_score_map_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/binary_score_map")
        if not os.path.exists(binary_score_map_path):
            os.makedirs(binary_score_map_path)

        gt_pred_map_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/gt_pred_score_map")
        if not os.path.exists(gt_pred_map_path):
            os.makedirs(gt_pred_map_path)

        mask_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/mask")
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        gt_pred_seg_image_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/gt_pred_seg_image")
        if not os.path.exists(gt_pred_seg_image_path):
            os.makedirs(gt_pred_seg_image_path)

        return score_map_path, binary_score_map_path, gt_pred_map_path, mask_path, gt_pred_seg_image_path

    def save_seg_results(self, scores, binary_scores, mask, name):
        score_map_path, binary_score_map_path, gt_pred_score_map, mask_path, gt_pred_seg_image_path = self.save_paths()
        img_name = name.split("/")
        img_name = "-".join(img_name[-2:])
        print(img_name)
        # score map
        imsave(
            os.path.join(score_map_path, "{}".format(img_name)),
            img_as_ubyte(np.clip(scores, 0, 1)),
            check_contrast=False,
        )

        # binary score map
        imsave(
            os.path.join(binary_score_map_path, "{}".format(img_name)),
            img_as_ubyte(binary_scores.astype(bool)),
            check_contrast=False,
        )

        # mask
        imsave(
            os.path.join(mask_path, "{}".format(img_name)),
            img_as_ubyte(np.clip(mask, 0, 1)),
            check_contrast=False,
        )

        # # pred vs gt map
        # imsave(os.path.join(gt_pred_score_map, "{}".format(img_name)), normalize(binary_scores + mask))
        visulization_score(img_file=name, mask_path=mask_path,
                     score_map_path=score_map_path, saving_path=gt_pred_score_map)
        # pred vs gt image
        visulization(img_file=name, mask_path=mask_path,
                     score_map_path=binary_score_map_path, saving_path=gt_pred_seg_image_path)

    @staticmethod
    def _serializable_config(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [AnoSegDFR._serializable_config(item) for item in value]
        if isinstance(value, list):
            return [AnoSegDFR._serializable_config(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): AnoSegDFR._serializable_config(item)
                for key, item in value.items()
            }
        return value

    def checkpoint_config(self):
        cfg = getattr(self, 'cfg', None)
        if cfg is None:
            return {}
        return {
            key: self._serializable_config(value)
            for key, value in vars(cfg).items()
        }

    def save_model(self, epoch=0):
        checkpoint_path = os.path.join(self.model_path, 'autoencoder.pth')
        temporary_path = checkpoint_path + '.tmp'
        cfg = getattr(self, 'cfg', None)
        checkpoint = {
            'checkpoint_version': 2,
            'epoch': epoch,
            'autoencoder': self.autoencoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'n_dim': self.n_dim,
            'config': self.checkpoint_config(),
            'seed': getattr(cfg, 'seed', None),
            'training_elapsed_seconds': self.training_elapsed_seconds,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        }
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        np.save(os.path.join(self.model_path, 'n_dim.npy'), self.n_dim)

    def load_training_checkpoint(self):
        model_path = os.path.join(self.model_path, 'autoencoder.pth')
        if not os.path.exists(model_path):
            return 0
        data = torch.load(model_path, map_location=self.device, weights_only=True)
        self.autoencoder.load_state_dict(data['autoencoder'])
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
        self.training_elapsed_seconds = float(
            data.get('training_elapsed_seconds', 0.0)
        )
        if 'torch_rng_state' in data:
            torch.set_rng_state(data['torch_rng_state'].cpu())
        if torch.cuda.is_available() and data.get('cuda_rng_state_all'):
            torch.cuda.set_rng_state_all([
                state.cpu() for state in data['cuda_rng_state_all']
            ])
        epoch = int(data.get('epoch', 0))
        print(f"Resuming {self.data_name} from epoch {epoch}: {model_path}")
        if (
            'config' not in data
            or 'seed' not in data
            or 'training_elapsed_seconds' not in data
        ):
            self.save_model(epoch)
            print(f"Upgraded legacy checkpoint metadata: {model_path}")
        return epoch

    def load_model(self, path=None):
        print("Loading model...")
        if path is None:
            model_path = os.path.join(self.model_path, 'autoencoder.pth')
            print("model path:", model_path)
            if not os.path.exists(model_path):
                print("Model not exists.")
                return False

            data = torch.load(model_path, map_location=self.device, weights_only=True)

            self.autoencoder.load_state_dict(data['autoencoder'])
            self.training_elapsed_seconds = float(
                data.get('training_elapsed_seconds', 0.0)
            )
            print("Model loaded:", model_path)
        return True

    # def save_dim(self):
    #     np.save(os.path.join(self.model_path, 'n_dim.npy'))

    def load_dim(self, model_path):
        dim_path = os.path.join(model_path, 'n_dim.npy')
        if not os.path.exists(dim_path):
            print("Dim not exists.")
            self.n_dim = None
        else:
            self.n_dim = np.load(os.path.join(model_path, 'n_dim.npy'))

    ########################################################
    #  Evaluation (testing)
    ########################################################
    def segmentation_results(self):
        def normalize(x):
            return x/x.max()

        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=self.threshold)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            if name[0].split("/")[-2] != "good":
                self.save_seg_results(normalize(scores), binary_scores, mask, name)
            # self.save_seg_results((scores-score_min)/score_range, binary_scores, mask, name)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))

    ######################################################
    #  Evaluation of segmentation
    ######################################################
    def save_segment_paths(self, fpr):
        # generating saving paths
        binary_score_map_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/fpr_{}/binary_score_map".format(fpr))
        if not os.path.exists(binary_score_map_path):
            os.makedirs(binary_score_map_path)

        mask_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/fpr_{}/mask".format(fpr))
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        gt_pred_seg_image_path = os.path.join(self.cfg.save_path+"/Results", self.subpath + "/fpr_{}/gt_pred_seg_image".format(fpr))
        if not os.path.exists(gt_pred_seg_image_path):
            os.makedirs(gt_pred_seg_image_path)

        return binary_score_map_path, mask_path, gt_pred_seg_image_path

    def save_segment_results(self, binary_scores, mask, name, fpr):
        binary_score_map_path, mask_path, gt_pred_seg_image_path = self.save_segment_paths(fpr)
        img_name = name.split("/")
        img_name = "-".join(img_name[-2:])
        print(img_name)
        # binary score map
        imsave(os.path.join(binary_score_map_path, "{}".format(img_name)), binary_scores)

        # mask
        imsave(os.path.join(mask_path, "{}".format(img_name)), mask)

        # pred vs gt image
        visulization(img_file=name, mask_path=mask_path,
                     score_map_path=binary_score_map_path, saving_path=gt_pred_seg_image_path)

    def estimate_thred_with_fpr(self, expect_fpr=0.05):
        """
        Use training set to estimate the threshold.
        """
        threshold = 0
        scores_list = []
        for i, normal_img in enumerate(self.train_data_loader):
            normal_img = normal_img[0:1].to(self.device)
            scores_list.append(self.score(normal_img).detach().cpu().numpy())
        scores = np.concatenate(scores_list, axis=0)

        # find the optimal threshold
        max_step = 100
        min_th = scores.min()
        max_th = scores.max()
        delta = (max_th - min_th) / max_step
        for step in range(max_step):
            threshold = max_th - step * delta
            # segmentation
            binary_score_maps = np.zeros_like(scores)
            binary_score_maps[scores <= threshold] = 0
            binary_score_maps[scores > threshold] = 1

            # estimate the optimal threshold base on user defined min_area
            fpr = binary_score_maps.sum() / binary_score_maps.size
            print(
                "threshold {}: find fpr {} / user defined fpr {}".format(threshold, fpr, expect_fpr))
            if fpr >= expect_fpr:  # find the optimal threshold
                print("find optimal threshold:", threshold)
                print("Done.\n")
                break
        return threshold

    def segment_evaluation_with_fpr(self, expect_fpr=0.05):
        # estimate threshold
        thred = self.estimate_thred_with_fpr(expect_fpr=expect_fpr)

        # segment
        i = 0
        metrics = []
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)
            scores, binary_scores = self.segment(img, threshold=thred)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_segment_results(binary_scores, mask, name, expect_fpr)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        print("threshold:", thred)

    def segment_evaluation_with_otsu_li(self, seg_method='otsu'):
        """
        ref: skimage.filters.threshold_otsu
        skimage.filters.threshold_li
        e.g.
        thresh = filters.threshold_otsu(image) #返回一个阈值
        dst =(image <= thresh)*1.0 #根据阈值进行分割
        """
        from skimage.filters import threshold_li
        from skimage.filters import threshold_otsu

        # segment
        thred = 0
        time_start = time.time()
        for i, (img, mask, name) in enumerate(self.test_data_loader):    # batch size is 1.
            i += 1

            # segment
            img = img.to(self.device)

            # estimate threshold and seg
            if seg_method == 'otsu':
                thred = threshold_otsu(img.detach().cpu().numpy())
            else:
                thred = threshold_li(img.detach().cpu().numpy())
            scores, binary_scores = self.segment(img, threshold=thred)

            mask = mask.squeeze().numpy()
            name = name[0]
            # save results
            self.save_segment_results(binary_scores, mask, name, seg_method)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time()-time_start))
        print("threshold:", thred)

    def segmentation_evaluation(self):
        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return
        self.segment_evaluation_with_fpr(expect_fpr=self.cfg.except_fpr)

    def validation(self, epoch):
        i = 0
        time_start = time.time()
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            i += 1
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # score
            score = self.score(img).detach().cpu().numpy()

            masks.append(mask)
            scores.append(score)
            print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(bool)
        scores = np.array(scores)

        # auc score
        auc_score, roc = auc_roc(masks, scores)
        # metrics over all data
        print("auc:", auc_score)
        out_file = os.path.join(self.eval_path, '{}_epoch_auc.csv'.format(self.model_name))
        if not os.path.exists(out_file):
            with open(out_file, mode='w') as f:
                f.write("Epoch" + ",AUC" + "\n")
        with open(out_file, mode='a+') as f:
            f.write(str(epoch) + "," + str(auc_score) + "\n")

    def metrics_evaluation(self, expect_fpr=0.3, max_step=5000, save_visualizations=True):
        from sklearn.metrics import auc
        from sklearn.metrics import roc_auc_score, average_precision_score
        from skimage import measure
        import pandas as pd

        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        print("Calculating AUC, IOU, PRO metrics on testing data...")
        time_start = time.time()
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # anomaly score
            # anomaly_map = self.score(img).data.cpu().numpy()
            anomaly_map = self.score(img).detach().cpu().numpy()

            masks.append(mask)
            scores.append(anomaly_map)
            if save_visualizations:
                score_range = anomaly_map.max() - anomaly_map.min()
                normalized = (anomaly_map - anomaly_map.min()) / score_range if score_range > 0 else np.zeros_like(anomaly_map)
                binary = (normalized > self.threshold).astype(np.uint8)
                self.save_seg_results(normalized, binary, mask, name[0])
            #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        scores = np.array(scores)
        
        # binary masks
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(bool)
        
        # auc score (image level) for detection
        labels = masks.any(axis=1).any(axis=1)
#         preds = scores.mean(1).mean(1)
        preds = scores.max(1).max(1)    # for detection
        det_auc_score = roc_auc_score(labels, preds)
        det_pr_score = average_precision_score(labels, preds)
        
        # auc score (per pixel level) for segmentation
        seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
        seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
        # metrics over all data
        print(f"Det AUC: {det_auc_score:.4f}, Seg AUC: {seg_auc_score:.4f}")
        print(f"Det PR: {det_pr_score:.4f}, Seg PR: {seg_pr_score:.4f}")
        
        # per region overlap and per image iou
        max_th = scores.max()
        min_th = scores.min()
        delta = (max_th - min_th) / max_step
        
        ious_mean = []
        ious_std = []
        pros_mean = []
        pros_std = []
        threds = []
        fprs = []
        binary_score_maps = np.zeros_like(scores, dtype=bool)
        for step in range(max_step):
            thred = max_th - step * delta
            # segmentation
            binary_score_maps[scores <= thred] = 0
            binary_score_maps[scores > thred] = 1

            pro = []    # per region overlap
            iou = []    # per image iou
            # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
            # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
            for i in range(len(binary_score_maps)):    # for i th image
                # pro (per region level)
                label_map = measure.label(masks[i], connectivity=2)
                props = measure.regionprops(label_map)
                for prop in props:
                    x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                    cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                    # cropped_mask = masks[i][x_min:x_max, y_min:y_max]   # bug!
                    cropped_mask = prop.image_filled
                    intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                    pro.append(intersection / prop.area)
                # iou (per image level)
                intersection = np.logical_and(binary_score_maps[i], masks[i]).astype(np.float32).sum()
                union = np.logical_or(binary_score_maps[i], masks[i]).astype(np.float32).sum()
                if masks[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                    iou.append(intersection / union)
            # against steps and average metrics on the testing data
            ious_mean.append(np.array(iou).mean())
#             print("per image mean iou:", np.array(iou).mean())
            ious_std.append(np.array(iou).std())
            pros_mean.append(np.array(pro).mean())
            pros_std.append(np.array(pro).std())
            # fpr for pro-auc
            masks_neg = ~masks
            fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
            fprs.append(fpr)
            threds.append(thred)
            
        # as array
        threds = np.array(threds)
        pros_mean = np.array(pros_mean)
        pros_std = np.array(pros_std)
        fprs = np.array(fprs)
        
        ious_mean = np.array(ious_mean)
        ious_std = np.array(ious_std)
        
        # save results
        data = np.vstack([threds, fprs, pros_mean, pros_std, ious_mean, ious_std])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std',
                                                        'ious_mean', 'ious_std'])
        # save results
        df_metrics.to_csv(os.path.join(self.eval_path, 'thred_fpr_pro_iou.csv'), sep=',', index=False)

        
        # best per image iou
        best_miou = ious_mean.max()
        print(f"Best IOU: {best_miou:.4f}")
        
        # default 30% fpr vs pro, pro_auc
        idx = fprs <= expect_fpr    # find the indexs of fprs that is less than expect_fpr (default 0.3)
        fprs_selected = fprs[idx]
        fprs_selected = rescale(fprs_selected)    # rescale fpr [0,0.3] -> [0, 1]
        pros_mean_selected = pros_mean[idx]    
        pro_auc_score = auc(fprs_selected, pros_mean_selected)
        print("pro auc ({}% FPR):".format(int(expect_fpr*100)), pro_auc_score)

        # save results
        data = np.vstack([threds[idx], fprs[idx], pros_mean[idx], pros_std[idx]])
        df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                        'pros_mean', 'pros_std'])
        df_metrics.to_csv(os.path.join(self.eval_path, 'thred_fpr_pro_{}.csv'.format(expect_fpr)), sep=',', index=False)

        # save auc, pro as 30 fpr
        with open(os.path.join(self.eval_path, 'pr_auc_pro_iou_{}.csv'.format(expect_fpr)), mode='w') as f:
                f.write("det_pr, det_auc, seg_pr, seg_auc, seg_pro, seg_iou\n")
                f.write(f"{det_pr_score:.5f},{det_auc_score:.5f},{seg_pr_score:.5f},{seg_auc_score:.5f},{pro_auc_score:.5f},{best_miou:.5f}")

        return {
            'det_pr': float(det_pr_score),
            'det_auc': float(det_auc_score),
            'seg_pr': float(seg_pr_score),
            'seg_auc': float(seg_auc_score),
            'seg_pro': float(pro_auc_score),
            'seg_iou': float(best_miou),
        }
            

    def metrics_detecion(self, expect_fpr=0.3, max_step=5000):
        from sklearn.metrics import auc
        from sklearn.metrics import roc_auc_score, average_precision_score
        from skimage import measure
        import pandas as pd

        if self.load_model():
            print("Model Loaded.")
        else:
            print("None pretrained models.")
            return

        print("Calculating AUC, IOU, PRO metrics on testing data...")
        time_start = time.time()
        masks = []
        scores = []
        for i, (img, mask, name) in enumerate(self.test_data_loader):  # batch size is 1.
            # data
            img = img.to(self.device)
            mask = mask.squeeze().numpy()

            # anomaly score
            # anomaly_map = self.score(img).data.cpu().numpy()
            anomaly_map = self.score(img).detach().cpu().numpy()

            masks.append(mask)
            scores.append(anomaly_map)
            #print("Batch {},".format(i), "Cost total time {}s".format(time.time() - time_start))

        # as array
        masks = np.array(masks)
        scores = np.array(scores)
        
        # binary masks
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1
        masks = masks.astype(bool)
        
        # auc score (image level) for detection
        labels = masks.any(axis=1).any(axis=1)
#         preds = scores.mean(1).mean(1)
        preds = scores.max(1).max(1)    # for detection
        det_auc_score = roc_auc_score(labels, preds)
        det_pr_score = average_precision_score(labels, preds)
        
        # auc score (per pixel level) for segmentation
        seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
        seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
        # metrics over all data
        print(f"Det AUC: {det_auc_score:.4f}, Seg AUC: {seg_auc_score:.4f}")
        print(f"Det PR: {det_pr_score:.4f}, Seg PR: {seg_pr_score:.4f}")
        
        # save detection metrics
        with open(os.path.join(self.eval_path, 'det_pr_auc.csv'), mode='w') as f:
                f.write("det_pr, det_auc\n")
                f.write(f"{det_pr_score:.5f},{det_auc_score:.5f}") 
            
