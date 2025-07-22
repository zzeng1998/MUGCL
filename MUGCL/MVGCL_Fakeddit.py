import sys,os
import logging
sys.path.append(os.getcwd())
#from ..Process.process import *
import torch
from torch_scatter import scatter_mean
import numpy as np
from optimization import BertAdam, warmup_linear
#from ..tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from rand5fold import *
from evaluate import *
from PIL import Image as II
from MeGCN_concat1 import BertForMeGCN,BertForGCN
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torchvision
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from train_val import *
torch.backends.cudnn.enabled = False
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

def load_fold_data(str):
    def read_label_txt(path):
        data = []
        f = open(path, 'r',encoding='utf-8')
        for line in f:
            line = line.strip()
            label, eid = line.split('\t')
            data.append(eid)
        f.close()
        return data

    train_path = os.path.join("/home/zz/MGFND/model/" + '/Dataset/fakeddit/'+'/'+str+'/train.label.txt')
    test_path = os.path.join("/home/zz/MGFND/model/" + '/Dataset/fakeddit/'+'/'+str+'/test.label.txt')
    return [read_label_txt(train_path),read_label_txt(test_path)]


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels),outputs

def image_reshape(img_name):
    file = '/home/zz/MGFND/Dataset/fakeddit/fakeddit/images_20000/'
    image = II.open(file + img_name).convert("RGB")
    image = image_transform(image)
    return image

def evaluation(label, pred):
    return accuracy_score(label, pred), f1_score(label, pred), precision_score(label, pred), recall_score(label,pred), roc_auc_score(label, pred)

def custom_collate(batch):
    # 对于每个元素在批次中，将Data对象转换为包含各个属性的字典
    batch_dict = {
        'x_input_ids': [item[0].x_input_ids for item in batch],
        'x_input_mask': [item[0].x_input_mask for item in batch],
        'x_segment_ids': [item[0].x_segment_ids for item in batch],
        'edge_index': [item[0].edge_index for item in batch],
        'image': [item[1].image.unsqueeze(0) for item in batch],
        'y': [item[0].y for item in batch]
    }

    # 返回字典形式的批次数据
    return batch_dict

class GraphDataset(Dataset):
    def __init__(self, fold_x, dataset_name, data_path=os.path.join("/home/zz/MGFND/Process" + '/dataset')):
        self.fold_x = fold_x
        self.data_path = os.path.join(data_path,dataset_name+'textgraph')


    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        image_id = str(id)+".jpg"
        image = image_reshape(image_id)
        # input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids,root=root_text,edgeindex=tree,rootindex=rootindex,y=y
        return Data(x_input_ids=torch.LongTensor([data['input_ids']]),
                    x_input_mask = torch.LongTensor([data['input_mask']]),
                    x_segment_ids = torch.LongTensor([data['segment_ids']]),
                    edge_index=torch.LongTensor(edgeindex),
                    y=torch.LongTensor([int(data['y'])])), Data(image = torch.tensor(image))


def train_fold(train_fold,test_fold):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prepare model
    model = BertForMeGCN.from_pretrained(bert_path,num_labels = num_labels)
    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = int(
            len(train_fold) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
    
    # train_losses,val_losses,train_accs,val_accs = [],[],[],[]
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list = GraphDataset(train_fold,dataset_name)
    testdata_list = GraphDataset(test_fold,dataset_name)
    train_loader = DataLoader(traindata_list, batch_size=train_batch_size, collate_fn=custom_collate,
                                shuffle=False, num_workers=0)
    test_loader = DataLoader(testdata_list, batch_size=eval_batch_size, collate_fn=custom_collate,
                                shuffle=False, num_workers=0)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    model.train()
    for epoch_ids in range(num_train_epochs):
            tr_loss = 0
            result_loss = 100
            acc_,pre_,rec_,f1_ = 0,0,0,0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
                
                input_ids, input_mask, segment_ids,edge,image, label_ids = batch["x_input_ids"],batch["x_input_mask"],batch["x_segment_ids"],batch["edge_index"],batch["image"],batch["y"]
                input_ids = torch.concat(input_ids,dim=0).to(device)
                input_mask = torch.concat(input_mask,dim=0).to(device)
                segment_ids = torch.concat(segment_ids,dim=0).to(device)
                label_ids = torch.concat(label_ids,dim=0).to(device)
                image = torch.concat(image,dim=0).to(device)
                edge = torch.concat(edge,dim=1).to(device)
                # batch = tuple(t.to(device) for t in batch)
                # input_ids, input_mask_tweet, segment_ids, label_ids = batch
                
                loss,_,_,_ = model(input_ids, segment_ids, input_mask,edge, image, label_ids)
                # if n_gpu > 1:
                #     loss = loss.mean() # mean() to average on multi-gpu.
                # if args.gradient_accumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps

                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y,prediction = [],[]
            feature_text = np.empty(shape=(0, 768))
            feature_image = np.empty(shape=(0, 768))
            feature_final = np.empty(shape=(0, 768))
            label_save = np.empty(shape=(0,1))
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids, input_mask, segment_ids, edge, image, label_ids = batch["x_input_ids"], batch[
                    "x_input_mask"], batch["x_segment_ids"], batch["edge_index"], batch["image"], batch["y"]
                input_ids = torch.concat(input_ids, dim=0).to(device)
                input_mask = torch.concat(input_mask, dim=0).to(device)
                segment_ids = torch.concat(segment_ids, dim=0).to(device)
                label_ids = torch.concat(label_ids, dim=0).to(device)
                image = torch.concat(image, dim=0).to(device)
                edge = torch.concat(edge, dim=1).to(device)

                with torch.no_grad():
                    tmp_eval_loss, image1, text, multimodal = model(input_ids, segment_ids, input_mask, edge, image,
                                                                   label_ids)
                    logits, _, _, _ = model(input_ids, segment_ids, input_mask, edge, image)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy, pre_ids = accuracy(logits, label_ids)
                y += label_ids.tolist()
                prediction += pre_ids.tolist()
                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

                feature_text = np.vstack((feature_text, text.detach().cpu().numpy()))
                feature_image = np.vstack((feature_image, image1.detach().cpu().numpy()))
                feature_final = np.vstack((feature_final, multimodal.detach().cpu().numpy()))
                label_save = np.array(y)

            if epoch_ids == 0:
                np.savetxt("feature_fakeddit/feature_text.txt", feature_text, fmt="%f", delimiter=" ")
                np.savetxt("feature_fakeddit/feature_image.txt", feature_image, fmt="%f", delimiter=" ")
                np.savetxt("feature_fakeddit/feature_final.txt", feature_final, fmt="%f", delimiter=" ")
                np.savetxt("feature_fakeddit/label.txt", label_save, fmt="%f", delimiter=" ")
            if epoch_ids == 7:
                np.savetxt("feature_fakeddit/feature_final.txt", feature_final, fmt="%f", delimiter=" ")
                np.savetxt("feature_fakeddit/label.txt", label_save, fmt="%f", delimiter=" ")

            Acc,Acc1, Prec1, Recll1, F1,Acc2, Prec2, Recll2, F2=evaluationclass(prediction,y)
            Accu, Pre, Rec, F11, AUC=evaluation(prediction,y)
            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss/nb_tr_steps
            if (eval_loss<result_loss):
                acc_,pre_,rec_,f1_ = Acc,Prec2,Recll2,F2
            result = {'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'global_step': global_step,
                    'loss': loss,
                    'acc':Acc,
                    'acc1':Acc1,
                    'pre1':Prec1,
                    'recall1':Recll1,
                    'f1':F1,
                    'acc2':Acc2,
                    'pre2':Prec2,
                    'recall2':Recll2,
                    'f2':F2,
                    'Accuracy':Accu,
                    'Pre':Pre,
                    'Rec':Rec,
                    'F1':F11,
                    'AUC':AUC}

            output_eval_file = os.path.join(output_dir, f"{fold5data_val}_eval_results{epoch_ids}.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** %s Eval results *****",str(epoch_ids))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            # Save a trained model
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # output_model_file = os.path.join(output_dir, f"pytorch_model{epoch_ids}.bin")
    return acc_,pre_,rec_,f1_
            #torch.save(model_to_save.state_dict(), output_model_file)
            

dataset_name = "fakeddit"

max_seq_length = 100
train_batch_size = 8
eval_batch_size = 8
learning_rate = 5e-5
num_train_epochs = 15
gradient_accumulation_steps = 1
output_dir = "/home/zz/MGFND/results/megcn/fakeddit"
bert_model = "bert-base-uncased"
num_labels = 2
bert_path = os.path.join("/home/zz/MGFND/Process/" + bert_model)
warmup_proportion = 0.1
patience = 7
seed = 42

fold5data = ["split_0","split_1","split_2","split_3","split_4"]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
acc,pre,rec,f1 = 0,0,0,0
for fold5data_val in fold5data:
    fold0_train,fold0_test = load_fold_data(fold5data_val)
    acc_,pre_,rec_,f1_ = train_fold(fold0_train,fold0_test)
    acc+=acc_
    pre+=pre_
    rec+=rec_
    f1+=f1_
result = {"acc:":acc/5,"pre:":pre/5,"rec:":rec/5,"f1:":f1/5}
output_eval_file = os.path.join(output_dir, f"results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** %s Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
