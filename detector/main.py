import torch
from torch import nn
from .params_mv4 import get_params
from .layers import ATSSAssocHead, VLFPN
from .layers import MV4VL
from .det_utils import pre_processing, post_processing
torch.set_grad_enabled(False)


class ATSSHumanFaceAssocDetector(nn.Module):
    def __init__(self,
                 human_score_thr=0.384,
                 face_score_thr=0.365,
                 assoc_score_thr=0.095,
                 assoc_iof_coeff=0.1,
                 use_gpu=False,
                 fp16=False):
        if fp16 and not use_gpu:
            raise TypeError('You can use fp16 only for gpu')
        super(ATSSHumanFaceAssocDetector, self).__init__()
        params = get_params()
        self.backbone = MV4VL(**params['backbone_params'])
        self.neck = VLFPN(**params['neck_params'])
        self.bbox_head = ATSSAssocHead(**params['head_params'])
        self.init_weights(params['weight_path'])
        self.human_score_thr, self.face_score_thr = human_score_thr, face_score_thr
        self.use_gpu, self.fp16, self.assoc_score_thr = use_gpu, fp16, assoc_score_thr
        self.assoc_iof_coeff = assoc_iof_coeff
        if use_gpu:
            self.cuda()
            if fp16:
                self.half()

    def init_weights(self, weight_path):
        sd = torch.load(weight_path, 'cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
        sd = {key: val for key, val in sd.items() if 'num_batches_tracked' not in key}
        self.load_state_dict(sd, strict=True)
        self.bbox_head.merge_reg_scale()
        self.backbone.merge_normalize()
        self.eval()

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        feats = self.bbox_head(feats)
        return feats

    def calibrate_score(self):
        dets_score_thrs = [self.human_score_thr, self.face_score_thr]
        assoc_score_thr = self.assoc_score_thr
        assert len(dets_score_thrs) == self.bbox_head.num_objects
        self.bbox_head.calibrate_score(dets_score_thrs, assoc_score_thr)

        self.assoc_iof_coeff = self.assoc_iof_coeff * (0.5 / self.assoc_score_thr)
        self.human_score_thr = self.face_score_thr = self.assoc_score_thr = 0.5

    @torch.no_grad()
    def predict(self, img_rgb):
        batch = pre_processing(img_rgb)
        if self.fp16:
            batch = batch.half()
        if self.use_gpu:
            batch = batch.cuda()
        feats = self.forward(batch)
        return post_processing(feats,
                               human_score_thr=self.human_score_thr,
                               face_score_thr=self.face_score_thr,
                               assoc_score_thr=self.assoc_score_thr,
                               assoc_iof_coeff=self.assoc_iof_coeff,
                               invert_assoc_score=self.bbox_head.calibrate)
