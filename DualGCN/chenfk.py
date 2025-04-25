import argparse
import torch

from models.ian import IAN
from models.atae_lstm import ATAE_LSTM
from models.syngcn import SynGCNClassifier
from models.semgcn import SemGCNClassifier
from models.dualgcn import DualGCNClassifier
from models.dualgcn_bert import DualGCNBertClassifier
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData


def main():
    model_classes = {
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'syngcn': SynGCNClassifier,
        'semgcn': SemGCNClassifier,
        'dualgcn': DualGCNClassifier,
        'dualgcnbert': DualGCNBertClassifier,
    }

    dataset_files = {
        'restaurant': {
            'train': './DualGCN/dataset/Restaurants_corenlp/train.json',
            'test': './DualGCN/dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train.json',
            'test': './dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './DualGCN/dataset/Tweets_corenlp/train.json',
            'test': './DualGCN/dataset/Tweets_corenlp/test.json',
        }
    }

    input_colses = {
        'atae_lstm': ['text', 'aspect'],
        'ian': ['text', 'aspect'],
        'syngcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj'],
        'semgcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length'],
        'dualgcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj'],
        'dualgcnbert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end',
                        'adj_matrix', 'src_mask', 'aspect_mask']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='dualgcn', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

    parser.add_argument('--attention_heads', default=1, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--vocab_dir', type=str,
                        default='D:\\Desktop\\DualGCN-ABSA-main\\DualGCN-ABSA-main\\DualGCN\\dataset\\Laptops_corenlp')
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
    parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--losstype', default="doubleloss", type=str,
                        help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--beta', default=0.2, type=float)

    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.post_size = 30
    opt.pos_size = 30

    tokenizer = build_tokenizer(
        fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
        max_length=opt.max_length,
        data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset))
    embedding_matrix = build_embedding_matrix(
        vocab=tokenizer.vocab,
        embed_dim=opt.embed_dim,
        data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

    model = DualGCNClassifier(embedding_matrix, opt).to("cuda")
    x = []
    for i in range(9):
        if i == 7:  # 第 7 个张量（索引 6），对应 l，形状 (16,)
            # 生成序列长度，范围为 1 到 max_seq_len（例如 85）
            l = torch.randint(1, 85, (16,)).long()
            x.append(l)
        else:  # 其他张量（tok, asp, pos, head, deprel, post, mask, adj），形状 (16, 85)
            # 生成整数张量，假设值为 0 到 100（可根据需求调整范围）
            tensor = torch.randint(0, 2000, (16, 85))
            x.append(tensor)
    y1,y2 = model(x)
    print(y1.shape)



if __name__ == '__main__':
    main()
