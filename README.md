# upstage-cv-classfication
Document Type Classification | 문서 타입 분류

# 1. 경진대회 소개

## 1.1. 개요

- Computer Vision 이미지 분류 대회

> 주어진 문서 이미지를 총 17종의 문서 유형으로 분류하는 대회
> 

## 1.2. 일정

- 대회 시작일: 2024.02.05
- 최종 제출 마감 기한: 2024.02.19 19:00

## 1.3. 평가 지표

- Macro F1 Score

## 1.4. 데이터 설명

- 학습 데이터
    - 총 17종(자동차 번호판, 자동차 계기판, 자동차 등록증, 계좌번호, 여권, 운전면허증, 주민등록증, 진단서, 처방전, 통원/진료 확인서, 입퇴원 확인서, 진료비 영수증, 약제비 영수증, 진료비 납입 확인서, 건강보험 임신출산 진료비 지급 신청서, 이력서, 소견서)의 1570장의 문서 이미지
    - 각 클래스별로 46 ~ 100장으로 구성
- 평가 데이터
    - 여러 augmentation이 적용되어 있는 3140장의 문서 이미지

# 2.  경진대회 수행 절차 및 방법

## 2. 1. 환경

- **컴퓨팅 환경:** 인당 RTX 3090 서버를 VSCode와 SSH로 연결하여 사용
- **협업 환경:** Github, Wandb
- **의사 소통:** Slack, Zoom

## 2.1. 수행 절차

- Step 1: 유사 경진대회 분석 및 인사이트 도출
- Step 2: EDA
- Step 3: Data Processing - Image Preprocessing, Augmentation
- Step 4: Modeling - 모델 선택, Hyper-parameter Tuning 및 Ensemble을 통한 성능 실험
- Step 5: 학습된 모델 성능을 기반으로 DL 파이프라인 반복
- Step 7: 최종 제출 파일 선택

## 2.2. 수행 방법

- 매일 Zoom 팀 미팅을 통해 진행상황 및 아이디어 공유
- Github repository를 사용해 작업 코드 공유
- WandB([https://wandb.ai/aistages-cv-04](https://wandb.ai/aistages-cv-04/projects))를 사용한 실험 결과 공유
- Slack을 통한 실시간 의견 교류

# 3. 경진대회 수행 과정

## 3.1. EDA

### 3.1.1. 이미지 크기 확인

- 학습 데이터
    - Width - Mean: 497.61 / STD: 79.35
    - Height - Mean: 538.17  / STD: 76.05
    
    ![train_size_v.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef8dbd9-414c-4cf5-813d-32ecb943cc67/b75adc76-a961-4181-9c72-a4d0a84d9b49/train_size_v.png)
    
- 평가 데이터
    - Width - Mean: 517.09 / STD: 79.83
    - Height - Mean: 518.55 / STD: 79.79
    
    ![test_size_v.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef8dbd9-414c-4cf5-813d-32ecb943cc67/ce1b9909-7f01-4b2d-9904-10c51a9d6974/test_size_v.png)
    

### 3.1.2. 클래스 분포

![class_dist.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef8dbd9-414c-4cf5-813d-32ecb943cc67/f72fd0bd-1dc6-4e0b-a696-3e1426e8a5c7/class_dist.png)

### 3.1.3. 이미지 시각화

- 학습 데이터
    - 모두 정방향의 노이즈 없는 이미지로 구성됨
- 학습 데이터
    - 다음과 같은 다양한 Augmentation이 적용되어 있는 것으로 파악
        1. Rotation
        2. Flip
        3. Rotation + Flip
        4. Noise / Patterned Noise
        5. Blur
        6. Mixup

## 3.2. Data Processing

### 3.2.1. 잘못된 레이블 파악

- 총 7개의 학습 이미지 레이블이 잘못된 것을 파악 → 올바른 레이블로 업데이트
    
    ![wrong_label_v.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/91aa266d-848f-4067-8e23-44f36786ad1a/wrong_label_v.png)
    

### 3.2.2. 이미지 전처리

- 사전 학습 모델에 사용된 이미지 크기로 Resize
    - 문서의 가로 세로 비율이 유지되도록 Padding 적용
- 사전 학습 모델에 사용된 정규화 값(평균 및 표준 편차)으로 정규화 적용

### 3.2.3 학습 데이터 Augmentation

- 학습 시간 단축을 위해 offline 방식으로 증강
- 랜덤 변형 적용하여 증강 횟수 늘려가며 실험 → 증강 이미지 개수(6만개 ~ 14만개) 늘어날수록 성능 향상
- 증강 기법
    - 평가 이미지를 최대한 유사하게 재현할 수 있는 증강 기법을 선택하여 적용
        1. 0도, 90도, 180도, 270도 회전(4) X 반전(2) => 8배 증강
            
            ![base_aug_v.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/3c955b17-6c26-42db-a4c3-c475099d58c0/base_aug_v.png)
            
        2. 노이즈 및 기타 변형 랜덤 적용
            - 가우시안 및 패턴 노이즈 적용
            - 블러 효과
            - 밝기, 색조 및 채도 변형
            - Grayscale 변환
            - Shift + Rotate
            
            ![aug_train_7_v.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/9f12b723-8da9-4f3c-a832-1c51ff2174d0/aug_train_7_v.png)
            
        
        > 사용 라이브러리
        - Albumentation: Transpose, HorizontalFlip VerticalFlip, Blur, GaussNoise, ToGray, ShiftScaleRotate 등
        - Augraphy: PatternGenerator
        > 
- 클래스간 불균형 해소를 위해 데이터 Oversampling
    - 클래스간 데이터 개수의 균형을 맞추기 위해 각 클래스별 가중치를 설정하여 데이터 증식
- Hold-Out 방식으로 학습/검증 데이터셋 분할

## 3.3. Modeling

### 3.3.1. 모델 선택

- resnet50, resnext50, efficientnet_b0, efficientnet_b4 pre-trained 모델로 실험
    
    ![스크린샷 2024-02-19 오후 10.07.05.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/cc361b61-d45f-4a52-8146-adc915a42d0d/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-02-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.07.05.png)
    
    > 입력 이미지 크기가 커질수록, 모델이 복잡할수록 최적화 성능 향상
    → 성능이 가장 좋았던 efficientnet_b4으로 고정
    > 

### 3.3.2. Modeling Process

- Optimizer - Adam
- Learning Rate - 0.001에서 0.0005로 조정
- Early Stopping
    - patience를 5로 설정하여 5 epoch 동안 validation loss가 감소하지 않으면 조기 종료

> 예측 결과 시각화
→ 양식이 유사한 3, 7, 14 클래스에 대한 예측 성능이 떨어진다는 것을 확인
> 
- 3, 7, 14 클래스에 대한 샘플링 가중치를 증가시킴 -> 성능 향상
- 3, 7, 14 클래스 별도 학습
    - efficientnet_b5 pre-trained 모델로 해당 클래스만 따로 학습하여 기존 결과값 대체 -> 스코어 향상
- efficientnet_b4의 classifier 블록(fc layer)에 간단한 attention mechanism을 구현
- Test-Time Augmentation 적용
    - inference 단계에서 평가 이미지에 Flip(반전), RandomRotate(90도 단위 랜덤 회전)을 적용해 online 방식으로 augmentation하여 N회 예측 수행
        - 20회 inference 후 soft-voting 앙상블 -> 스코어 향상
- 리더보드 기준 최상위 예측값들을 hard-voting으로 앙상블
-> 최종 리더보드 Public 스코어 0.9631 달성

# 4. 경진대회 결과

## 4.1. 리더보드 순위

- Public 리더보드
    - 4위, F1 score: 0.9631

![스크린샷 2024-02-22 오후 9.34.41.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/8f7c0186-063c-44b9-9a30-c1035f5e23d5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-02-22_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.34.41.png)

- Private(최종) 리더보드
    - 3위, F1 score: 0.9547

![스크린샷 2024-02-22 오후 9.34.26.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/47e828b4-2141-400f-812f-099aa08a6672/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-02-22_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.34.26.png)

## 4.2. 결과 분석

### 스코어 향상에 도움이 된 시도 및 전략

1. 평가 이미지와 유사한 학습 이미지 구축을 위한 다양한 Augmentation 기법 적용
    - 평가 이미지에 적용된 Augmentation을 최대한 빠짐없이 학습 이미지에 재현하기 위해 평가 이미지를 시각화하여 *어떤 변형 기법들이 적용되었는지 파악*하고 *Albumentation 및 Augraphy 라이브러리의 해당 이미지 변형 기능을 찾아 각 기법에 대한 적절한 적용 비율을 선택*하여 학습 이미지 Augmentation에 활용했다.
2. 양식이 유사한 클래스 간의 예측 성능 개선을 위한 전략 시도
    - CNN 모델에 Attention Mechanism 적용
        - 모델이 중요한 부분에 더 강한 가중치를 부여하여 성능을 개선할 수 있도록 *CNN 기반의 Efficientnet_b4 모델에 간단한 Attention 레이어를 추가한 Custom 모델을 구축*했다.
    - 모델이 혼동하기 쉬운 클래스 오버샘플링 및 별도 학습
        - 모델의 예측 결과 시각화를 통해 3, 7, 14 클래스를 서로 혼동하여 예측하는 경우가 많다는 것을 파악하였고 *해당 클래스에 대해 샘플링 가중치를 증가*시켜 dataloader가 해당 클래스를 오버샘플링하도록 했다.
        - 양식이 비슷해 클래스 간 *혼동을 야기하는 클래스끼리 별도로 학습*한 후 기존 학습 결과 해당 클래스로 예측한 값을 별도 학습한 모델의 예측값으로 대체했다.
3. Test-Time Augmentation 적용
    - 테스트 시 입력 이미지에 Augmentation을 적용하면 특정 방향으로 회전되거나 반전된 이미지에 대한 언더피팅으로 인한 예측력 저하가 개선됨과 동시에 모델의 일반화 성능이 향상될 것으로 판단했다.
        - 따라서 Inference 단계에서 *평가 이미지에* *Flip(반전), RandomRotate(90도 단위 랜덤 회전) 기법으로 Online Augmentation하여 20회 예측을 수행한 뒤 결과값을 Soft-Voting 방식으로 앙상블*했다.

### **마주한 한계 및 아쉬웠던 점**

1. 더 복잡한 모델을 사용하지 못한 것
    - GPU 메모리 부족으로 인해 Vision Transformer 기반 모델과 같은 더 복잡한 모델을 사용해 학습하는 데 제약이 있었고, 메모리 사용량을 줄이기 위해 배치사이즈를 줄일 경우 학습 시간이 너무 길어지는 문제가 있었다. 이같은 제약을 극복하기 위해 메모리 사용 효율성을 극대화할 수 있는 모델 경량화 기법들을 시도해보지 못한 것이 아쉽다.
2. 생각했던 방법론을 모두 시도해보지 못한 것
    - 머신러닝에 비해 모델 학습에 소요되는 시간이 길어 실험을 충분히 수행하지 못했다. 추가적인 실험을 통해 다양한 전략들을 탐색하고 시도해볼 수 있었다면 더 높은 스코어를 기록할 수 있었을 것이라고 생각된다.
