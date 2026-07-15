# 설명 자료 (HTML)

브라우저로 열어 보는 자기완결(self-contained) 설명 페이지. 외부 의존성 없음(이미지 임베드), 한/영 전환 지원.

- [matching_process.html](matching_process.html) — **Autochecker 매칭 프로세스**. RoMa 2D 매칭 → depth gate → 3D RANSAC 상대포즈 → anchor(L/R/U) 추정 → safe zone 검사 흐름. rerun 시각화 캡처 임베드.
- [anchor_invariance.html](anchor_invariance.html) — **카메라가 움직여도 템플릿은 그대로**. 런타임 카메라가 바뀌어도(intrinsic + cam_to_base 만 주어지면) teaching 템플릿 재작성 없이 anchor 를 계속 추정할 수 있는 원리. safe zone 은 base(로봇) 프레임 절대 위치라 카메라 이동과 무관.

> tail(orbbec)·hood(photoneo) 공통 개념 자료. 프로토콜 차이는 각 `docs/` 체크리스트/Protocols 참고.
