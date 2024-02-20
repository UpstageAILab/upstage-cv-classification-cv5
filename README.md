[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/3DbKuh4a)
# Document Type Classification Competitions
## Team

|![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9233ab6e-25d5-4c16-8dd4-97a7b8535baf) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/e7394268-0f94-4468-8cf5-3cf67e4edd07) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9c75cbd9-f409-4fdd-a5c3-dec082ade3bf) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/388eac05-7cd9-4688-8a87-5b6b742715cf) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/48dd674c-ab93-48d1-9e05-e7e8e402597c) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/0a524747-a854-4eee-95b6-108c84514df8) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [최장원](https://github.com/UpstageAILab)             |            [김영천](https://github.com/UpstageAILab)             |            [배창현](https://github.com/UpstageAILab)             |            [박성우](https://github.com/UpstageAILab)             |            [조예람](https://github.com/huB-ram)             |            [이소영B](https://github.com/UpstageAILab)             |
|                            팀장                            |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |

## 1. Competitions Info
### Overview
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/4e685524-05b9-4f48-b980-d24460bb43cb)

### Environment
Vscode, RTX 3090 server

### Timeline

- February 05, 2024 - Start Date
- February 19, 2024 - Final submission deadline

### Evaluation
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/1c2bc659-2d35-4678-9a54-6a6671e002c8)

## 2. Components

### Directory

```
├── code
    ├── EDA
    ├── Augmentation
    ├── Modelling
    └── Ensemble
```

## 3. Data descrption

### Dataset overview

이번 대회는 computer vision domain에서 가장 중요한 태스크인 이미지 분류 대회입니다.

이미지 분류란 주어진 이미지를 여러 클래스 중 하나로 분류하는 작업입니다. 이러한 이미지 분류는 의료, 패션, 보안 등 여러 현업에서 기초적으로 활용되는 태스크입니다. 딥러닝과 컴퓨터 비전 기술의 발전으로 인한 뛰어난 성능을 통해 현업에서 많은 가치를 창출하고 있습니다.
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/65c0f5ec-08f0-4fa6-b176-e57c2cc8868d)

그 중, 이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용될 데이터는 총 17개 종의 문서로 분류되어 있습니다. 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/8a55ea9c-8165-4521-831c-f97ec5621729)

이번 대회를 통해서 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다. computer vision에서 중요한 backbone 모델들을 실제 활용해보고, 좋은 성능을 가지는 모델을 개발할 수 있습니다. 그 밖에 학습했던 여러 테크닉들을 적용해 볼 수 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

### EDA & Augmentaion
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/42d2efe5-a159-412e-9d2f-fc0af56da09c)
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/11816180-a75c-49c6-b0f0-37bf804f8968)

### Augmentation
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/962f162a-a445-4b60-9a1f-de6d2757db3b)



## 4. Modeling

### Model descrition
metaformer: https://github.com/sail-sg/metaformer?tab=readme-ov-file


### Modeling Process
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/a3c67807-fca2-46c3-b6b0-f763b9a1807f)  
.  
.  
.  
.  
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/c0d76c59-c669-4204-ae00-c859f9f2dbd3)  
```python
# training 코드, evaluation 코드, training_loop 코드
def training(model, dataloader, optimizer, loss_fn, scheduler, device, epoch, num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    train_loss = 0.0
    preds_list = []
    targets_list = []
    
    m = torch.nn.Softmax(dim=-1)

    tbar = tqdm(dataloader)
    for idx, (image, targets) in enumerate(tbar):
        image = image.to(device)
        targets = targets.to(device)
        
        # 순전파
        model.zero_grad(set_to_none=True)
        if (idx + 1) % 10 == 0:
            image, mix_targets = mixup_fn(image, targets)
            preds = model(image)
            loss = mixup_loss_fn(preds, mix_targets)
        else:
            preds = model(image)
            # loss = loss_fn(preds, targets)
            loss = loss_fn(m(preds), targets)

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 손실과 정확도 계산
        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())
        tbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}")

    # 에폭별 학습 결과 출력
    train_loss = train_loss / len(dataloader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')
    
    ret = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return model, ret

def evaluation(model, dataloader, loss_fn, device, epoch, num_epochs):
    model.eval()  # 모델을 평가 모드로 설정
    valid_loss = 0.0
    preds_list = []
    targets_list = []
    m = torch.nn.Softmax(dim=-1)

    with torch.no_grad(): # model의 업데이트 막기
        tbar = tqdm(dataloader)
        for idx, (image, targets) in enumerate(tbar):
            image = image.to(device)
            targets = targets.to(device)

            # 순전파
            model.zero_grad(set_to_none=True)
            if (idx + 1) % 8 == 0:
                image, mix_targets = mixup_fn(image, targets)
                preds = model(image)
                loss = mixup_loss_fn(preds, mix_targets)
            else:
                preds = model(image)
                loss = loss_fn(m(preds), targets)

            # 손실과 정확도 계산
            valid_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())
            tbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Valid Loss: {loss.item():.4f}")
            
    # 에폭별 학습 결과 출력
    valid_loss = valid_loss / len(dataloader)
    valid_acc = accuracy_score(targets_list, preds_list)
    valid_f1 = f1_score(targets_list, preds_list, average='macro')
    
    ret = {
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "valid_f1": valid_f1,
    }

    return model, ret


def training_loop(model, train_dataloader, valid_dataloader, trn_dataset, val_dataset, train_dataset_list, val_dataset_list, loss_fn, optimizer, scheduler, device, num_epochs, patience, filename, project_name):
    best_valid_loss = float('inf')  # 가장 좋은 validation loss를 저장
    early_stop_counter = 0  # 카운터
    valid_max_acc = -1
    valid_max_f1 = -1
    
    notes = f"Optimizer: {optimizer.__class__.__name__}, focal_gamma: {loss_fn.gamma}, Image_Size: {img_size},"
    run = wandb.init(project = project_name, tags=[optimizer.__class__.__name__], notes=notes)
    
    for epoch in range(num_epochs):
        
        data_idx = epoch % 200
        T_dataset = ConcatDataset([trn_dataset, train_dataset_list[data_idx]])
        V_dataset = ConcatDataset([val_dataset, val_dataset_list[data_idx]])
        
        train_dataloader = DataLoader(T_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
        valid_dataloader = DataLoader(V_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
        print(f"dataset{data_idx}")
        
        model, train_ret = training(model, train_dataloader, optimizer, loss_fn, scheduler, device, epoch, num_epochs)
        model, valid_ret = evaluation(model, valid_dataloader, loss_fn, device, epoch, num_epochs)
        
        monitoring_value = {'train_loss': train_ret['train_loss'], 'train_accuracy': train_ret['train_acc'], 'train_f1': train_ret['train_f1'], 
                            'valid_loss': valid_ret['valid_loss'], 'valid_accuracy': valid_ret['valid_acc'], 'valid_f1': valid_ret['valid_f1'], 
                            'best_valid_loss': best_valid_loss, 'valid_max_acc': valid_max_acc, 'valid_max_f1': valid_max_f1,
                            'init_lr': optimizer.param_groups[0]['initial_lr'], 'lr': optimizer.param_groups[0]['lr'],
                            'weight decay': optimizer.param_groups[0]['weight_decay'],}
        
        run.log(monitoring_value, step=epoch)
        
        if valid_ret['valid_acc'] > valid_max_acc:
            valid_max_acc = valid_ret['valid_acc']
            
        if valid_ret['valid_f1'] > valid_max_f1:
            name, ext = os.path.splitext(filename)
            torch.save(model.state_dict(), model_path+name+'_f1'+ext)
            valid_max_f1 = valid_ret['valid_f1']
        
        # validation loss가 감소하면 모델 저장 및 카운터 리셋
        if valid_ret['valid_loss'] < best_valid_loss:
            best_valid_loss = valid_ret['valid_loss']
            # model.save_pretrained(model_path)
            name, ext = os.path.splitext(filename)
            torch.save(model.state_dict(), model_path+name+'_loss'+ext)
            early_stop_counter = 0
            
        # validation loss가 증가하거나 같으면 카운터 증가
        else:
            early_stop_counter += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], T_Train Loss: {train_ret['train_loss']:.4f}, Train Accuracy: {train_ret['train_acc']:.4f}, Train F1: {train_ret['train_f1']:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], T_Valid Loss: {valid_ret['valid_loss']:.4f}, Valid Accuracy: {valid_ret['valid_acc']:.4f}, Valid F1: {valid_ret['valid_f1']:.4f}")

        # 조기 종료 카운터가 설정한 patience를 초과하면 학습 종료
        if early_stop_counter >= patience:
            print("Early stopping")
            break
            
    run.finish()
    return model, valid_max_acc, valid_max_f1
```
```python
# 모델 전체 fine tuning
model.to(device)
num_epochs = EPOCHS
lr = LR
patience = 25
filename = 'caformer_s18_sail_in22k_ft_in1k_384.pth'
project_name = "kyc-DC-caformer_s18_386"

# optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

num_epoch_steps = len(trn_loader) * 2
# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/1000, max_lr=lr, step_size_up=10, step_size_down=50, mode='exp_range', gamma=0.995)
# scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=num_epoch_steps*80, T_mult=2, eta_max=lr,  T_up=num_epoch_steps*1, gamma=0.5)

# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=LR, step_size_up=4, step_size_down=10, mode='triangular')
scheduler = get_scheduler(
    name='cosine', optimizer=optimizer, 
    num_warmup_steps=0, 
    num_training_steps= num_epoch_steps * num_epochs
)

model, valid_max_acc, valid_max_f1 = training_loop(model, trn_loader, val_loader, trn_dataset, val_dataset, aug_trn_dataset_list, aug_val_dataset_list, loss_fn, optimizer, scheduler, device, num_epochs, patience, filename, project_name)
print(f'Valid max accuracy : {valid_max_acc:5f}, Valid max f1 : {valid_max_f1:5f}')
```
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/3953836d-4b55-47bb-b25f-8d5a14f523b1)
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/04220b2b-eca5-4ac0-9f5b-18a82f9d3441)
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/e45f2a46-7144-412e-a7ac-9aed3a534dac)

## Ensemble & TTA
https://github.com/qubvel/ttach?tab=readme-ov-file
https://github.com/qubvel/ttach/blob/master/ttach/wrappers.py #L52

## 5. Result

### Leader Board
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/5e233a57-3e37-4141-885c-220fd5273e59)  
최종 리더보드  
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/4df686f9-9b6d-484c-9c6c-c3c115f3dbfa)


### Presentation
https://docs.google.com/presentation/d/1BEArU4iON6gzoHsSwt1XaK0Dg4vGRUF3cxYcQWTcDvI/edit#slide=id.g269915ddf82_4_15

### Reference
https://paperswithcode.com/  
https://github.com/qubvel/ttach?tab=readme-ov-file  
https://github.com/sail-sg/metaformer
