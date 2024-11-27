import os  
import matplotlib.pyplot as plt  
import numpy as np  
from .metrics import metric


#绘制test结果对比图
def draw_comparasion(args,trues,preds,ii):
    print(preds.shape,trues.shape)
    preds_sample = preds[-1]
    trues_sample = trues[-1]
    print('draw shape',preds_sample.shape,trues_sample.shape)
    # if args.features == 'MS':
    #     preds_sample = preds_sample[:,-1]
    #     trues_sample = trues_sample[:,-1]

    print('draw shape',preds_sample.shape,trues_sample.shape)
    mae_sample, mse_sample, rmse_sample, mape_sample, mspe_sample, smape_sample, nd_sample,r2_sample = metric(preds_sample, trues_sample)

    # 创建一个绘图
    plt.figure(figsize=(12, 6))


    # 绘制 preds 和 trues 的曲线
    plt.plot(preds_sample, label='Predictions', alpha=0.7)
    plt.plot(trues_sample, label='True Values', alpha=0.7)
    # print('Predictions vs True Values feature: '+ str(feat_ids[random_index,0]))
    plt.title('Predictions vs True Values feature: ')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend(title=f'MAE: {mae_sample:.4f}\nMSE: {mse_sample:.4f}\nR^2: {r2_sample:.4f}')

    result_folder = './results/predict_images/'+args.data_path.split('.')[0]
    if not os.path.exists(result_folder):        os.makedirs(result_folder)
    # 保存图像
    print(result_folder+'/'+args.model+'_'+args.data_path.split('.')[0]+'_'+args.features + '_'+str(args.pred_len)+'_'+str(ii)+'.jpg')
    plt.savefig(result_folder+'/'+args.model+'_'+args.data_path.split('.')[0]+'_'+args.features + '_'+str(args.pred_len)+'_'+str(ii)+'.jpg')

#绘制收敛图
def draw_losses(args,train_losses,vali_losses,test_losses,ii):
    # 绘制loss图像
    plt.figure(figsize=(10, 5))
    # 绘制训练损失
    plt.plot(train_losses, label='Train Loss', color='red', linewidth=2)
    # 绘制验证损失
    plt.plot(vali_losses, label='Validation Loss', color='green', linewidth=2)
    # 绘制测试损失
    plt.plot(test_losses, label='Test Loss', color='blue', linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses over Epochs')
    plt.legend()
    plt.grid(True)

    result_folder = './results/loss_images/' + args.data_path.split('.')[0]
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 保存图像
    print(result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_loss.png')
    plt.savefig(result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_loss.png', dpi=300, format='png')

# 保存预测结果
def store_results(args,trues,preds,ii):
    result_folder = './results/result_files/'+args.data_path.split('.')[0]

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # 获取数组的shape
    pred_shape = preds.shape
    true_shape = trues.shape
    
    preds_file_name = result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_preds.npy'
    trues_file_name = result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_trues.npy'
    shape_file = result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_shape.txt'

    # 将shape写入txt文件
    with open(shape_file, 'w') as f:
        f.write(f'preds shape: {pred_shape}\n')
        f.write(f'trues shape: {true_shape}\n')
    
    np.save(preds_file_name, preds)
    np.save(trues_file_name, trues)

def store_str(args,str):
    result_folder = './results/mae_mses/'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)


    file_name = result_folder + '/'+args.data_path.split('.')[0]+'_'+args.model+'_' + args.target +'_'+args.features+'.txt'

    with open(file_name,'a') as f:
        f.write(str+'\n')
    print(str)
