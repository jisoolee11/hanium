![](image_files/bg_img.jpg)

# Calorie for Diet
> 바코드 및 음식물 사진을 통하여 음식을 인식하고 음식의 무게 정보를 받아 정확한 영양정보를 계산하는 서비스입니다.


<br/>

# 팀 소개

: 서예선 - 프론트엔드 

: 이지수 - 백엔드

: 이채은 - 프론트엔드

: 최정은 - 백엔드

<br/>

# 개발 방법
- 음식물 인식에 yolov3 모델을 사용했습니다.

- 바코드 인식과 카메라에 opencv를 사용했습니다.

- 무게측정과 무선 통신을 사용하기 위해 :grapes: 라즈베리파이와 로드셀, hx711을 사용했습니다.

<br/>

# 사용 방법 & 결과

1. yolov3.weights 파일을 다운 받아 폴더에 넣기
2. `. venv/scripts/activate` 로 가상환경 실행
3. 데이터베이스 실행 `flask db migrate`
4. `flask db upgrade`
5. `export FLASK_APP=__init__` 환경변수 설정
6. `flask run` 실행


<br/>

# 업데이트

* 0.2.1
    * 수정: 문서 업데이트!
* 0.2.0
    * 추가: `init()` 메서드 추가
* 0.1.1
    * 수정: `macOS` 에서 실행 안되는 현상 수정
* 0.1.0
    * 대망의 첫 출시!
* 0.0.1
    * Repo init!

