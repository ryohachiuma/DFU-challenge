from ..builder import DETECTORS
from .two_stage import TwoStageDetector

class EnsembleFaster(BaseDetector):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def simple_test(self, img, img_meta, **kwargs):
        pass

    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    def extract_feat(self, imgs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rpn_test_cfg = self.models[0].test_cfg.rpn
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for model in self.models:
            # recompute feats to save memory
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                proposal_list = model.simple_test_rpn(x, img_meta, rpn_test_cfg)
                for i, proposals in enumerate(proposal_list):
                    aug_proposals[i].append(proposals)
        # after merging, proposals will be rescaled to the original image size
        proposal_list = [
            merge_aug_proposals(proposals, img_meta, rpn_test_cfg)
            for proposals, img_meta in zip(aug_proposals, img_metas)
        ]

        rcnn_test_cfg = self.models[0].test_cfg.rcnn
        aug_bboxes = []
        aug_scores = []
        aug_img_metas = []
        for model in self.models:
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                # only one image in the batch
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']

                proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                         scale_factor, flip)
                # "ms" in variable names means multi-stage
                ms_scores = []

                rois = bbox2roi([proposals])
                for i in range(model.num_stages):
                    bbox_head = model.bbox_head[i]
                    cls_score, bbox_pred = model._bbox_forward_test(i, x, rois)
                    ms_scores.append(cls_score)

                    if i < model.num_stages - 1:
                        bbox_label = cls_score.argmax(dim=1)
                        rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                          img_meta[0])

                cls_score = sum(ms_scores) / float(len(ms_scores))
                bboxes, scores = model.bbox_head[-1].get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=False,
                    cfg=None)
                aug_bboxes.append(bboxes)
                aug_scores.append(scores)
                aug_img_metas.append(img_meta)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, aug_img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels, self.models[0].bbox_head[-1].num_classes)

        return bbox_result
