import argparse

# Genearte arguments for train, test
def get_opts():
    parser = argparse.ArgumentParser(description='Active Metric Learning for Exposure Set Retrieval')

    ## Paths and device
    parser.add_argument('--data_dir', type=str, default='/datadrive/jianx/data/train_data/ance_training_rank100_nqueries50000_200000_Sep_03_22:56:31.csv',
                        help='data folder')
    parser.add_argument('--pretrain_model_path', type=str, default='/datadrive/ruohan/rerank/train_query_50000_morepos/reverse_alpha0.5_layer1_residual1000_100_1000_0.0001_768.model',
                        help='pretrained model path')    
    parser.add_argument('--out_dir', type=str, default='/datadrive/ruohan/final_models/',
                        help='output folder')
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='device')    

    ## training data settings
    parser.add_argument('--num_query', type=int, default=50000,
                        help='Number of training queries')
    parser.add_argument('--num_passage', type=int, default=200000,
                        help='Number of training passages')
    parser.add_argument('--active_learning_stage', type=str, default="no_active",
                        help='active learning option')       

    ## Network settings
    parser.add_argument('--network_type', type=str, default='append',
                        help='Network type') 
    parser.add_argument('--pretrained_option', type=str, default="No",
                        help='Pretrained or not')                            
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Sampling parameter alpha') 
    parser.add_argument('--top_k', type=int, default=100,
                        help='Top k')       
    parser.add_argument('--epoch_size', type=int, default=100,
                        help='Epoch size')   
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Minibatch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate') 
    # default for append
    parser.add_argument('--embed_size', type=int, default=32,
                        help='Embedding size')
    parser.add_argument('--num_hidden_nodes', type=int, default=64,
                        help='Number of hidden nodes')  
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                        help='Number of hidden layers')   
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')     

    # testing
    parser.add_argument('--test_data_path', type=str, default='/datadrive/ruohan/final_train_test_data/ance_testing_rank100_nqueries50000_npassages20000.csv',
                        help='Test data folder')
    parser.add_argument('--test_output_path', type=str, default='/datadrive/ruohan/final_evaluation/',
                        help='Test output folder')    
    parser.add_argument('--reverse_ranker_path', type=str, default='/datadrive/ruohan/fix_residual_overfit/reverse_alpha0.5_layer1_residual1000_100_1000_0.0001_768.model',
                        help='Reverse ranker path')  

    # evaluation
    parser.add_argument('--eval_data_path', type=str, default='/datadrive/jianx/data/train_data/ance_testing_rank100_nqueries50000_20000_Sep_03_22:56:31.csv',
                        help='Evaluation results path') 

    ##
    opts = parser.parse_args()
    return opts

# Specific arguments for active learning
def get_opts_active_learning():
    parser = argparse.ArgumentParser(description='Active Metric Learning for Exposure Set Retrieval')  

    parser.add_argument('--active_learning_option', type=str, default="No",
                        help='Active Learning or not')
    parser.add_argument('--active_learning_stage', type=str, default='no_active',
                        help='Active learning stage')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--reverse_ranker_path', type=str, default='/datadrive/ruohan/final_models/no_active_residual_50000_query_200000_passage.model',
                        help='Reverse ranker path')
    opts = parser.parse_args()
    return opts


