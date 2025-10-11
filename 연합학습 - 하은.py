"""
연합학습(Federated Learning) 예제 코드
- FedAvg / FedProx / FedBN 세 가지 집계 방식을 비교 실험
- 각 데이터셋(MNIST, SVHN, USPS, SynthDigits, MNIST-M)을 개별 클라이언트로 설정
- 서버는 각 클라이언트로부터 모델 업데이트를 받아 평균 후 다시 배포
"""

# ==============================
# (0) 기본 설정 및 모듈 임포트
# ==============================
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 상위 경로
sys.path.append(base_path)  # 로컬 패키지(nets, utils 등) 임포트 가능하게 경로 추가

import torch
from torch import nn, optim
import time
import copy
from nets.models import DigitModel            # 숫자 인식용 소형 CNN 모델(사용자 정의)
import argparse
import numpy as np
import torchvision.transforms as transforms
from utils import data_utils                  # 커스텀 데이터셋 로더(DigitsDataset)

# ==============================
# (1) 데이터 준비: 각 도메인을 '클라이언트' 데이터로 사용
# ==============================
def prepare_data(args):
    # ------------------------------------------
    # 각 데이터셋(도메인)에 맞는 전처리(Transform) 구성
    # → 이미지 크기, 채널 수, 정규화 범위 등을 통일
    # ------------------------------------------
    transform_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),   # 원래 1채널(흑백) MNIST 이미지를 3채널(RGB)로 확장
        transforms.ToTensor(),                         # [0,255] 범위를 [0,1]로 변환 (Tensor 타입)
        transforms.Normalize((0.5, 0.5, 0.5),          # 평균 0.5, 표준편차 0.5로 정규화 (픽셀값 -1~1 범위로)
                             (0.5, 0.5, 0.5))
    ])

    transform_svhn = transforms.Compose([
        transforms.Resize([28, 28]),                   # SVHN(32x32)을 28x28 크기로 리사이즈
        transforms.ToTensor(),                         # Tensor로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 동일한 정규화 적용
    ])

    transform_usps = transforms.Compose([
        transforms.Resize([28, 28]),                   # USPS(16x16)을 28x28로 리사이즈
        transforms.Grayscale(num_output_channels=3),   # 1채널을 3채널로 확장 (모델 입력 차원 맞추기)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_synth = transforms.Compose([
        transforms.Resize([28, 28]),                   # SynthDigits를 28x28로 크기 통일
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnistm = transforms.Compose([
        transforms.ToTensor(),                         # MNIST-M은 이미 3채널 → 크기 변환 불필요
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ------------------------------------------
    # 각 도메인(=클라이언트)의 학습/테스트 데이터셋 구성
    # percent=args.percent : 전체 데이터 중 일부만 샘플링 비율로 사용
    # train=True → 학습용 / False → 테스트용
    # ------------------------------------------
    mnist_trainset = data_utils.DigitsDataset("../data/MNIST", 1, args.percent, True,  transform_mnist)
    mnist_testset  = data_utils.DigitsDataset("../data/MNIST", 1, args.percent, False, transform_mnist)

    svhn_trainset  = data_utils.DigitsDataset("../data/SVHN", 3, args.percent, True,  transform_svhn)
    svhn_testset   = data_utils.DigitsDataset("../data/SVHN", 3, args.percent, False, transform_svhn)

    usps_trainset  = data_utils.DigitsDataset("../data/USPS", 1, args.percent, True,  transform_usps)
    usps_testset   = data_utils.DigitsDataset("../data/USPS", 1, args.percent, False, transform_usps)

    synth_trainset = data_utils.DigitsDataset("../data/SynthDigits/", 3, args.percent, True,  transform_synth)
    synth_testset  = data_utils.DigitsDataset("../data/SynthDigits/", 3, args.percent, False, transform_synth)

    mnistm_trainset = data_utils.DigitsDataset("../data/MNIST_M/", 3, args.percent, True,  transform_mnistm)
    mnistm_testset  = data_utils.DigitsDataset("../data/MNIST_M/", 3, args.percent, False, transform_mnistm)

    # ------------------------------------------
    # DataLoader : 배치 단위로 데이터를 공급하는 객체
    # - batch_size=args.batch → 한 번에 불러올 데이터 크기
    # - shuffle=True → 학습용 데이터는 매 epoch마다 섞기 (일반화 향상)
    # - shuffle=False → 테스트 데이터는 순서 고정 (재현성 유지)
    # ------------------------------------------
    mnist_train_loader  = torch.utils.data.DataLoader(mnist_trainset,  batch_size=args.batch, shuffle=True)
    mnist_test_loader   = torch.utils.data.DataLoader(mnist_testset,   batch_size=args.batch, shuffle=False)

    svhn_train_loader   = torch.utils.data.DataLoader(svhn_trainset,   batch_size=args.batch, shuffle=True)
    svhn_test_loader    = torch.utils.data.DataLoader(svhn_testset,    batch_size=args.batch, shuffle=False)

    usps_train_loader   = torch.utils.data.DataLoader(usps_trainset,   batch_size=args.batch, shuffle=True)
    usps_test_loader    = torch.utils.data.DataLoader(usps_testset,    batch_size=args.batch, shuffle=False)

    synth_train_loader  = torch.utils.data.DataLoader(synth_trainset,  batch_size=args.batch, shuffle=True)
    synth_test_loader   = torch.utils.data.DataLoader(synth_testset,   batch_size=args.batch, shuffle=False)

    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch, shuffle=True)
    mnistm_test_loader  = torch.utils.data.DataLoader(mnistm_testset,  batch_size=args.batch, shuffle=False)

    # ------------------------------------------
    # 각 클라이언트(=도메인)가 고유 데이터로 학습하도록 리스트에 저장
    # → 각 데이터셋은 분포가 다르므로 비IID(Non-Identical) 환경 구성
    #   즉, 클라이언트 간 데이터 분포가 다름 → 연합학습의 현실적 조건
    # ------------------------------------------
    train_loaders = [
        mnist_train_loader,
        svhn_train_loader,
        usps_train_loader,
        synth_train_loader,
        mnistm_train_loader
    ]

    test_loaders = [
        mnist_test_loader,
        svhn_test_loader,
        usps_test_loader,
        synth_test_loader,
        mnistm_test_loader
    ]

    # 학습용 / 테스트용 DataLoader 리스트 반환
    return train_loaders, test_loaders



# ==============================
# (2) 로컬 학습(클라이언트) 함수
# ==============================
def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()                                  # 모델을 학습 모드로 전환(BN/Dropout 활성화)
    num_data, correct, loss_all = 0, 0, 0          # 누적 샘플 수, 정답 수, 손실 합계 초기화
    train_iter = iter(train_loader)                # DataLoader를 이터레이터로 변환
                                                   # (주의: 일반적으로는 len(train_loader) 사용이 더 안전)

    for step in range(len(train_iter)):            # 이터레이터 길이만큼 배치 반복
        optimizer.zero_grad()                      # 이전 배치에서 누적된 기울기 초기화
        x, y = next(train_iter)                    # 다음 배치(입력 x, 정답 y) 가져오기
        num_data += y.size(0)                      # 이번 배치 크기만큼 전체 샘플 수 누적
        x, y = x.to(device).float(), y.to(device).long()  # 입력/라벨을 디바이스로 이동 및 dtype 맞추기

        output = model(x)                          # 순전파: 모델 출력(로짓) 계산, shape [B, num_classes]
        loss = loss_fun(output, y)                 # 손실 계산(예: CrossEntropyLoss)
        loss.backward()                            # 역전파: 각 파라미터의 기울기 계산
        loss_all += loss.item()                    # 이번 배치 손실을 파이썬 수로 누적
        optimizer.step()                           # 옵티마이저로 파라미터 업데이트(SGD/Adam 등)

        pred = output.data.max(1)[1]               # 가장 큰 로짓의 인덱스(예측 클래스) 추출
        correct += pred.eq(y.view(-1)).sum().item()# 예측=정답인 개수만큼 누적

    return loss_all / len(train_iter), correct / num_data  # 배치 평균 손실, 전체 정확도 반환


def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    """
    FedProx: 기본 손실에 정규화항 (μ/2 * ||w - w0||^2)을 더해
    로컬 모델 파라미터(w)가 서버 전역모델(w0)에서 과도하게 벗어나지 않도록 제어.
    """
    model.train()                                  # 학습 모드로 전환
    num_data, correct, loss_all = 0, 0, 0          # 통계 변수 초기화
    train_iter = iter(train_loader)                # DataLoader → 이터레이터

    for step in range(len(train_iter)):            # 배치 반복
        optimizer.zero_grad()                      # 기울기 초기화
        x, y = next(train_iter)                    # 배치 로드
        num_data += y.size(0)                      # 전체 샘플 수 누적
        x, y = x.to(device).float(), y.to(device).long()  # 디바이스/자료형 정리

        output = model(x)                          # 순전파
        loss = loss_fun(output, y)                 # 기본 분류 손실

        # (중요) server_model은 외부(전역)에 선언된 전역모델
        # FedProx: 전역모델(server_model)과 로컬모델(model)의 파라미터 L2 거리 제곱을 벌점으로 추가
        if step > 0:                               # 첫 배치에서는 제외(초기 불안정/계산량 감소 목적)
            w_diff = torch.tensor(0., device=device)         # 스칼라 누적용 텐서(디바이스 일치)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)  # 파라미터별 L2노름 제곱을 누적 (||w-w0||^2)
            loss += args.mu / 2. * w_diff                    # μ/2 계수로 가중해 최종 손실에 더함

        loss.backward()                            # 역전파
        loss_all += loss.item()                    # 손실 누적
        optimizer.step()                           # 파라미터 업데이트

        pred = output.data.max(1)[1]               # 예측 클래스 인덱스
        correct += pred.eq(y.view(-1)).sum().item()# 정답 개수 누적

    return loss_all / len(train_iter), correct / num_data    # 배치 평균 손실, 전체 정확도 반환


# ==============================
# (3) 평가(테스트)
# ==============================
def test(model, test_loader, loss_fun, device):
    model.eval()                                  # 평가 모드로 전환 (BatchNorm, Dropout 고정)
                                                   # → 학습 때와 달리 배치 통계가 고정되고, 랜덤 드롭아웃 비활성화됨

    test_loss, correct = 0, 0                     # 전체 손실 누적용 변수와 정답 개수 초기화

    # -------------------------------------------------
    # 테스트 데이터셋 전체를 한 배치씩 평가
    # -------------------------------------------------
    for data, target in test_loader:              # test_loader는 (입력, 라벨) 배치 단위로 반복 제공
        data, target = data.to(device).float(), target.to(device).long()  
                                                   # GPU/CPU로 이동 및 dtype 맞추기
                                                   # float() → 입력값 실수형 변환
                                                   # long() → 라벨을 정수형으로 변환 (CrossEntropyLoss 요구사항)

        output = model(data)                      # 순전파(forward): 모델이 예측한 로짓(logit) 출력 (크기 [batch, classes])

        test_loss += loss_fun(output, target).item()  
                                                   # 손실 계산 후 파이썬 수(float)로 변환하여 누적
                                                   # .item()은 텐서를 스칼라 값으로 바꾸는 함수

        pred = output.data.max(1)[1]              # 각 샘플별로 예측 확률이 가장 큰 클래스 인덱스 추출 (argmax)
        correct += pred.eq(target.view(-1)).sum().item()  
                                                   # 예측(pred)과 실제(target)가 일치한 개수를 세서 누적

    # -------------------------------------------------
    # 전체 테스트 데이터에 대한 평균 손실과 정확도 계산
    # -------------------------------------------------
    avg_loss = test_loss / len(test_loader)       # 배치 단위 평균 손실
    accuracy = correct / len(test_loader.dataset) # 전체 데이터 대비 정확도 (% 아님, 0~1 범위)

    return avg_loss, accuracy                     # (평균 손실, 정확도) 반환


# ==============================
# (4) 서버 집계 (Aggregation)
# ==============================
def communication(args, server_model, models, client_weights):
    """
    모든 클라이언트의 모델 파라미터를 서버가 평균하여 업데이트하는 단계.
    - FedAvg / FedProx: 모든 파라미터를 단순 평균 (BatchNorm 추적 카운터 제외)
    - FedBN           : BatchNorm 파라미터(BN)는 로컬 유지, 나머지만 평균
    """
    with torch.no_grad():  # 파라미터 업데이트 중에는 그래디언트 추적 불필요 (메모리 절약)
        if args.mode.lower() == 'fedbn':  # FedBN 방식인 경우
            # ----------------------------------------------------
            # FedBN: 각 클라이언트의 BatchNorm 파라미터는 따로 유지
            #        (각 도메인별 통계가 달라서 BN 평균은 오히려 성능 저하를 유발)
            # ----------------------------------------------------
            for key in server_model.state_dict().keys():   # 모델의 모든 파라미터 이름(key)에 대해 반복
                if 'bn' not in key:                        # BN 관련 파라미터(bn1.weight, bn1.bias 등)는 제외
                    # 서버 파라미터와 동일한 크기의 0 텐서 생성 (임시 평균 저장용)
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)

                    # 모든 클라이언트의 해당 파라미터를 가중 평균
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]

                    # 계산된 평균값을 서버 모델에 복사 (서버 전역모델 갱신)
                    server_model.state_dict()[key].data.copy_(temp)

                    # 업데이트된 서버 파라미터를 다시 각 클라이언트 모델에 복사 (동기화)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        else:
            # ----------------------------------------------------
            # FedAvg / FedProx 방식:
            # 모든 파라미터를 평균하되, BatchNorm의 추적 카운터(num_batches_tracked)는 예외 처리
            # ----------------------------------------------------
            for key in server_model.state_dict().keys():   # 모델의 모든 파라미터(key)에 대해 반복
                if 'num_batches_tracked' in key:            # BatchNorm의 내부 통계 변수 (int형, 평균 불가)
                    # 첫 번째 클라이언트(models[0])의 값을 그대로 사용하여 통일
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    # 평균 계산용 임시 텐서 생성
                    temp = torch.zeros_like(server_model.state_dict()[key])

                    # 각 클라이언트 파라미터의 가중 평균 수행
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]

                    # 서버 모델 업데이트
                    server_model.state_dict()[key].data.copy_(temp)

                    # 서버 모델의 파라미터를 다시 각 클라이언트 모델로 복사 (브로드캐스트)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    # ----------------------------------------------------
    # 업데이트가 완료된 서버 모델과 클라이언트 모델 리스트 반환
    # ----------------------------------------------------
    return server_model, models

# ==============================
# (5) 메인: 연합학습 전체 루프
# ==============================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # GPU 사용 가능하면 CUDA, 아니면 CPU
    seed = 1                                                                # 재현성 확보용 시드
    np.random.seed(seed)                                                    # numpy 시드 고정
    torch.manual_seed(seed)                                                 # PyTorch 시드 고정
    torch.cuda.manual_seed_all(seed)                                        # 멀티 GPU일 때 모든 GPU 시드 고정
    print('Device:', device)                                                # 사용 디바이스 출력

    parser = argparse.ArgumentParser()                                      # 커맨드라인 인자 파서 생성
    parser.add_argument('--log', action='store_true')                       # 로그 파일 저장 여부 플래그
    parser.add_argument('--test', action='store_true')                      # 스냅샷 로드 후 테스트만 수행하는 모드
    parser.add_argument('--percent', type=float, default=0.1)               # 각 데이터셋 사용 비율(샘플링)
    parser.add_argument('--lr', type=float, default=1e-2)                   # 학습률
    parser.add_argument('--batch', type=int, default=32)                    # 배치 크기
    parser.add_argument('--iters', type=int, default=100)                   # 통신 라운드 수(aggregation 횟수)
    parser.add_argument('--wk_iters', type=int, default=1)                  # 라운드 내 로컬 학습 반복 횟수
    parser.add_argument('--mode', type=str, default='fedbn',                # 집계 방식 선택
                        help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2)                   # FedProx 정규화 계수 μ
    parser.add_argument('--save_path', type=str, default='../checkpoint/digits')  # 체크포인트 저장 루트
    parser.add_argument('--resume', action='store_true')                    # 이전 체크포인트에서 이어서 학습
    args = parser.parse_args()                                              # 인자 파싱

    exp_folder = 'federated_digits'                                         # 실험 폴더명
    args.save_path = os.path.join(args.save_path, exp_folder)               # 최종 저장 위치: save_path/실험폴더

    # FedAvg/FedBN/FedProx의 개념 요약(설명용 주석)
    # FedAvg: 모든 파라미터 평균 → 단일 전역모델
    # FedBN : BN 파라미터는 로컬 유지 → 도메인별 BN 통계 유지
    # FedProx: FedAvg + (μ/2)||w-w0||^2 → 전역모델에서 과도한 이탈 억제

    # 로깅 옵션
    log = args.log                                                          # 로그 기록 여부 플래그
    if log:
        log_path = os.path.join('../logs/digits/', exp_folder)              # 로그 저장 폴더 경로
        os.makedirs(log_path, exist_ok=True)                                # 폴더 없으면 생성
        logfile = open(os.path.join(log_path, f'{args.mode}.log'), 'a')     # 모드별 로그 파일 append 모드로 열기
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))  # 타임스탬프 기록
        logfile.write(f'===Setting===\n lr: {args.lr}\n batch: {args.batch}\n iters: {args.iters}\n wk_iters: {args.wk_iters}\n')  # 설정 기록

    # 저장 폴더 준비
    os.makedirs(args.save_path, exist_ok=True)                              # 체크포인트 저장 폴더가 없으면 생성
    SAVE_PATH = os.path.join(args.save_path, f'{args.mode}')                # 체크포인트 파일 경로(모드명으로 저장)

    # 서버 전역모델 초기화 및 손실함수 정의
    server_model = DigitModel().to(device)                                  # 전역모델(초기) 생성 후 디바이스로 이동
    loss_fun = nn.CrossEntropyLoss()                                        # 분류용 손실 함수

    # 데이터 준비
    train_loaders, test_loaders = prepare_data(args)                        # 각 도메인(클라이언트)별 데이터로더 생성
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']          # 클라이언트 이름 목록

    # 클라이언트 구성
    client_num = len(datasets)                                              # 클라이언트 수 = 도메인 수
    client_weights = [1 / client_num for _ in range(client_num)]            # 균등 가중치(필요시 크기/품질로 조정 가능)

# ==========================================
# [① Client Size 기반 skew 완화 지점]
#  - 예: 각 클라이언트 데이터 개수로 client_weights 재계산
#  - temp = [len(loader.dataset) for loader in train_loaders]
#  - client_weights = [x/sum(temp) for x in temp]
# ==========================================

    models = [copy.deepcopy(server_model).to(device) for _ in range(client_num)]  # 전역모델 복제 → 클라이언트 초기 파라미터

    # 이어서 학습 또는 테스트 전용 모드 처리
    if args.test:                                                          # --test가 켜진 경우
        print('Loading snapshots...')                                      # 스냅샷 로딩 안내
        checkpoint = torch.load('../snapshots/digits/{}'.format(args.mode.lower()))  # 미리 저장된 스냅샷 로드
        server_model.load_state_dict(checkpoint['server_model'])           # 서버 전역모델 파라미터 복원
        if args.mode.lower() == 'fedbn':                                   # FedBN은 클라이언트별 BN 분리 저장
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint[f'model_{client_idx}'])  # 각 클라 모델 복원
            for test_idx, test_loader in enumerate(test_loaders):          # 각 도메인에서 개별 모델로 테스트
                _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print(f'{datasets[test_idx]:<11s}| Test Acc: {test_acc:.4f}')
        else:                                                              # FedAvg/FedProx는 서버 모델 하나로 평가
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(f'{datasets[test_idx]:<11s}| Test Acc: {test_acc:.4f}')
        exit(0)                                                            # 테스트만 수행하고 종료

    if args.resume:                                                        # 체크포인트에서 재시작 옵션
        checkpoint = torch.load(SAVE_PATH)                                 # 마지막 저장 지점 로드
        server_model.load_state_dict(checkpoint['server_model'])           # 서버 전역모델 복원
        if args.mode.lower() == 'fedbn':                                   # FedBN이면 클라별 모델도 복원
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint[f'model_{client_idx}'])
        else:                                                              # 그 외(FedAvg/Prox)는 서버 파라미터로 동기화
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint.get('a_iter', 0)) + 1                 # 다음 라운드 인덱스 계산
        print(f'Resume training from epoch {resume_iter}')                 # 이어서 시작한다는 로그
    else:
        resume_iter = 0                                                    # 처음부터 시작

    # ==============================
    # (5-1) 학습 루프: 통신 라운드 반복
    # ==============================
    for a_iter in range(resume_iter, args.iters):                          # 전체 통신 라운드 반복
        optimizers = [optim.SGD(models[idx].parameters(), lr=args.lr)      # 각 클라이언트별 옵티마이저 생성(SGD)
                      for idx in range(client_num)]
        for wi in range(args.wk_iters):                                    # 라운드 내 로컬 학습을 반복(wk_iters 회)
            print(f"============ Train epoch {wi + a_iter * args.wk_iters} ============")  # 진행 로그
            if log:
                logfile.write(f"============ Train epoch {wi + a_iter * args.wk_iters} ============\n")

            # 각 클라이언트 로컬 학습
            for client_idx in range(client_num):                           # 모든 클라이언트에 대해
                model, train_loader, optimizer = (                         # 해당 클라의 모델/데이터/옵티마이저 선택
                    models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                )
                if args.mode.lower() == 'fedprox' and a_iter > 0:          # FedProx이고 첫 라운드 이후라면
                    train_fedprox(args, model, train_loader, optimizer,    # Prox 정규화 포함한 로컬 학습
                                  loss_fun, client_num, device)
                else:
                    train(model, train_loader, optimizer, loss_fun,        # 일반 로컬 학습(FedAvg/FedBN 또는 0라운드)
                          client_num, device)

# -------------------------------
# 서버 집계 단계 전 (가중치 조정 등 개입 포인트)
# -------------------------------

# ==========================================
# [② Label Distribution / Participation Skew 완화 지점]
#  - (Label Skew) 각 클라이언트의 라벨 다양성/편향 측정 후 client_weights 보정
#  - (Participation Skew) 최근 참여 빈도에 따라 가중치 보정
#  - 보정 후: client_weights = normalize(client_weights)
# ==========================================

        # 서버 집계
        server_model, models = communication(args, server_model, models, client_weights)  # 선택한 모드에 따라 파라미터 평균/브로드캐스트

        # 학습 데이터 성능 출력(모니터링)
        for client_idx in range(client_num):
            model = models[client_idx]                                   # 클라 모델 선택
            train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device)  # 학습셋으로 점검
            print(f'{datasets[client_idx]:<11s}| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            if log:
                logfile.write(f'{datasets[client_idx]:<11s}| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n')

        # 테스트 데이터 성능 출력(각 도메인 일반화 성능)
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)  # 도메인별 자체 모델로 평가(FedBN 가정)
            print(f'{datasets[test_idx]:<11s}| Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}')
            # 참고: FedAvg/Prox일 경우엔 server_model로 모든 도메인을 평가하도록 바꿔도 됨

    # (선택) 라운드 종료 후 체크포인트 저장은 별도 블록에서 처리 가능
    # 예) FedBN: 클라별 모델 + 서버 모델 저장 / FedAvg/Prox: 서버 모델만 저장
    #  -> 앞서 대화에서 다룬 저장 코드 블록을 여기 끝부분에 추가하세요.
