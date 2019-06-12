import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class NodeLookup(object):

#     label_look_up_path = "inception_model/imagenet_2012_challenge_label_map_proto.pbtxt"
    def __init__(self):
        label_look_up_path = "inception_model/imagenet_2012_challenge_label_map_proto.pbtxt"
        uid_lookup_path = "inception_model/imagenet_synset_to_human_label_map.txt"
        self.node_lookup = self.load(label_look_up_path, uid_lookup_path)

    def load(self, label_look_up_path, uid_lookup_path):
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        for line in proto_as_ascii_lines:
#             去掉换行
            line = line.strip('\n')
#             按照tab分割
            prase_items = line.split('\t')
#             编号
            uid = prase_items[0]
#             名称
            human_string = prase_items[1]
            uid_to_human[uid] = human_string
    
        proto_as_acsii = tf.gfile.GFile(label_look_up_path).readlines()
#     空字典
        node_id_to_uid = {}
        for line in proto_as_acsii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]
                
        node_id_to_name = {}
        
#         连接两个键值对
        for key, val in node_id_to_uid.items():
            name = uid_to_human[val]
            node_id_to_name[key] = name
        return node_id_to_name
    
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb','rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
#     遍历目录
# dirs子目录为空
    for root ,dirs,files in os.walk('images/'):
        for file in files:
            
#tensorflow载入图片
            image_data=tf.gfile.FastGFile(os.path.join(root,file),'rb').read()
            predictions=sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
            predictions=np.squeeze(predictions)
            
#             打印图片路径名称
            image_path=os.path.join(root,file)
            print(image_path)
            
#             显示图片
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            
            
#             排序          
            top_k=predictions.argsort()[-5:][::-1]
            node_lookup=NodeLookup()
            for node_id in top_k:
#                 获取名称
                human_string=node_lookup.id_to_string(node_id)
#                 获取分类置信度
                score=predictions[node_id]
                
                print('%s (score = %.5f)' %(human_string,score))
            print()















