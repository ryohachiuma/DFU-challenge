import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyDataset(CustomDataset):

    CLASSES = ('Ulcer', )

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)[1:]

        data_infos = []
        i = 0
        while i != len(ann_list) - 1:
            boxes = []
            labels = []
            while i != len(ann_list) - 1:
                anns = ann_list[i].split(',')
                if len(anns) == 6:
                    if float(anns[5]) < 0.9:
                        i+=1
                        continue
                filename = anns[0]
                boxes.append([float(anns[1]), float(anns[2]), float(anns[3]), float(anns[4])])
                labels.append(0)
                if i == len(ann_list) - 1 or filename != ann_list[i+1].split(',')[0]:
                    i+=1
                    break
                else:
                    i+=1
            #print('{}, {}'.format(filename, np.array(boxes)))
            _dict = dict(
                    filename=filename,
                    width=640,
                    height=480,
                    ann=dict(
                        bboxes=np.array(boxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                )
            data_infos.append(_dict)
        
        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']