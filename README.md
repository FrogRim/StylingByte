# StylingByte
코디 AI의 기초 알고리즘 및 모델링 구현

# File Tree
```bash
├── _data
│   ├── styles.csv -> 의류 정보 데이터 셋  
│   └── user.csv -> 사용자 정보(나이,성별) 데이터 셋
├── json  
│   └── recommended_clothes_with_data.json - ai_cody_result.py의 리턴 값으로 이것을 LoadJsonData.cs 코드를 통해 Unity환경으로 불러옴 
├── model
│   ├── age_model_test.h5 -> utkface_data_age_gender_predict.py 파일을 통해 만들어진 나이 판별 ai모델
│   ├── gender_model_test.h5 -> facerecognition_for_age_gender.py 파일을 통해 만들어진 성별 판별 ai모델
│   └── ncf_model.h5 -> 위에 두 모델을 이용해 얻은 사용자 정보와 의류 정보 데이터 셋을 이용해 NCF를 이용한 추천 딥러닝 모델 구축
├── picture
│   └── test_picture.pmg -> test case input data, 이 사진을 age_model과 gender_model을 통해 사진 속 인물의 성별과 나이 추출 후 알맞는 의류 추천
│   
│   
├── src_python
│   ├── utkface_data_age_gender_predict.py -> CNN을 이용하여 사용자의 사진을 보고 나이를 판별하는 모델
│   ├── facerecognition_for_age_gender.py -> CNN을 이용하여 사용자의 사진을 보고 성별을 판별하는 모델
│   └── ai_cody_result.py ->  NFC구현 모델, 그리고 그 후 한 번 더 협업 필터링을 통해 상위 3개의 추천모델 필터링하고 그 결과값을 JSON파일로 반환한다.
├── src_unity
│   ├── LoadJsonData.cs -> json파일의 형식을 규정한 class를 포함하며, 받은 json안의 data들을 읽어 class로 변환시킨 뒤, 유니티 로그에 출력
│   └── PythonRunner.cs -> UNITY환경에서 LOCAL에 세팅된 파이썬 Process를 실행시켜 pyrhon코드들을 전부 실행시킨 뒤, LoadJsonData.cs를 호출해 그 결과값을 Unity 프로세스로 가져온다.
│   
└── 
``` 
