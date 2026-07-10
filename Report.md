# 오류 보고서

## 요약
이 프로젝트는 적층제조에서 열확산을 모델링하기 위한 PINN 기반 코드이지만, 현재 상태에서는 학습 실행 단계에서 바로 문제가 될 가능성이 높은 지점들이 확인되었습니다.

## 확인된 주요 문제

### 1. train.py와 loss.py의 인터페이스 불일치
- [train.py](train.py) 에서는 `ResidualScales`를 import하고 `PINNLoss(..., scales=scales)` 형태로 호출합니다.
- 그러나 [loss.py](loss.py) 에는 `ResidualScales` 정의가 없고, `PINNLoss.__init__`도 `scales` 인자를 받지 않습니다.
- 결과적으로 실행 시점에 인자 불일치 또는 import 오류가 발생할 가능성이 큽니다.

### 2. 손실 스케일링이 구현되지 않음
- [train.py](train.py) 는 스케일링된 잔차를 사용하려는 구조로 보이지만, [loss.py](loss.py) 는 단순한 MSE 기반으로만 손실을 계산합니다.
- PDE, BC, 데이터 손실은 단위와 크기가 서로 다르므로, 이 상태로는 학습이 불안정할 가능성이 높습니다.

### 3. 경계조건 식의 물리적 부호 검토 필요
- [loss.py](loss.py) 의 top/surrounding boundary loss는 열유속 경계조건을 반영하려고 하지만, outward normal 부호와 방출/대류항의 배치를 다시 확인해야 합니다.
- 적층제조에서는 표면 경계조건이 매우 민감하므로, 부호가 틀리면 학습 결과가 물리적으로 잘못될 수 있습니다.

### 4. model.py의 정규화 버퍼 타입 문제
- [model.py](model.py) 에서 `self.coord_mean`, `self.coord_scale`, `self.branch_mean`, `self.branch_scale`를 버퍼로 등록한 후, forward에서 이 값들을 Tensor로 사용하고 있습니다.
- 현재 정적 분석 결과, 이들 값이 `Tensor | Module`로 오인되어 산술 연산이 실패하는 타입 오류가 보고되었습니다.
- 즉, 코드 실행 전 타입/구조상 확인이 필요합니다.

## 우선 수정 권장 순서
1. [loss.py](loss.py) 에 `ResidualScales`를 정의하고 `PINNLoss`가 `scales`를 받아서 사용할 수 있도록 수정
2. [train.py](train.py) 와 [loss.py](loss.py) 의 호출/반환 형식을 일치시킴
3. 경계조건 residual 식을 물리적으로 다시 검증
4. [model.py](model.py) 의 정규화 관련 코드와 타입 문제를 수정

## 결론
현재 코드base는 “구조는 거의 갖춰졌지만, 실행 가능한 상태로 맞물리지 않은” 상태입니다. 특히 손실 함수 인터페이스와 모델 입력 처리부가 먼저 정리되어야 실제 학습이 가능해집니다.
