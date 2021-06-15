# Hyperparameter Deep Ensembles
* "Hyperparameter Ensembles for Robustness and Uncertainty Quantification" 논문 리뷰 입니다.
* 출처: https://github.com/google/uncertainty-baselines

## 범위
* Hyperparameter Deep Ensemble
* 모델: wideresnet-28-10
* 데이터셋: cifar10

## 구성
* cpu: 베이스 라인 코드를 CPU로 가동시킬 수 있게 수정한 코드
* datasets: 데이터셋을 가져오고 전처리하는데 사용된 코드들과 설명
* models: 모델을 구축하는데 사용된 코드와 설명
* pipeLine: HyperparameterDeepEnsembles 구현 pipleLine
* saveCheckpoint: 단일 모델 학습과 하이퍼 파라미터 딥 앙상블을 구현할 때 사용하는 체크 포인트를 저장하는데 사용되는 코드와 설명

## HyperparameterDeepEnsembles 구현 pipeLine
* HyperparameterDeepEnsembles 구현 pipleLine은 다음과 같습니다.
  1. saveCheckpoint에 있는 코드를 작동시키면 모델이 학습을 진행하며 Checkpoints를 세이브 합니다.
  2. 저장된 Checkpoints를 로드시켜 ~를 수행 및 앙상블 사이즈를 조절하며 최적의 모델을 산출합니다.

## 한계
* 논문에서 제시된 베이스 라인 코드는 TPU나 GPU를 이용해 분산 처리를 하여 계산하도록 짜여 있습니다. 우리는 Colab과 같은 클라우드 환경을 이용해 TPU를 사용할 수 있습니다. 하지만 그럴 경우 모델이 사용한 매개변수를 저장하는 값인 Checkpoint의 디렉토리도 TPU 환경의 저장소에 저장해야 한다는 이슈가 발생합니다. 그 이슈를 해결하기 위해선 구글 클라우드 플랫폼을 사용하여 TPU 가상머신을 가동시켜야 하는데 비용문제 때문에 베이스 라인 코드를 CPU로 가동시킬 수 있게 코드를 수정하고 진행하였습니다.
* 하지만 CPU로 코드를 실행 시키니 렘의 사용량이 클라우드 환경의 한계 사용량을 넘는 문제가 생겨 결과를 직접 reproduce 하지 못했습니다.
