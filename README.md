# CAPTCHA_reader

기존 제안서와 프로젝트 주제가 완전히 달라졌습니다. <br>
colab에서 .ipynb 형식의 파일을 이용해 코드를 작성하였습니다.<br>
.ipynb을 사용하였기 때문에 소스코드에 주석과 텍스트 부분으로 코드에 대한 설명이 작성되어 있습니다.<br>

### 동기
먼저 기존 주제의 경우 오픈소스의 활용도가 생각보다 낮았으며, 수업 중 인공지능에 대해서 배웠던 만큼 이를 활용한 프로젝트를 진행하고자 하여 프로젝트의 주제를 완전히 바꾸기로 결정하였습니다. 먼저 떠올린 것은 손글씨를 인식하는 인공지능입니다. 이런 인공지능이 있는 것처럼 이를 활용한다면 변형된 이미지에 대해서도 인식하는 인공지능을 만들 수 있을 것이라고 생각했습니다. 그리고 이런 변형된 이미지의 경우 대표적으로 캡차 이미지가 있는데 이 캡차 이미지를 읽을 수 있다면 크롤링 과정을 할 때 데이터 수집에 좀 더 도움이 될 수 있을 것입니다.

이 상황에 대해 좀 더 자세히 설명하자면 프로젝트를 진행하기 위해 데이터를 얻는 과정에서 크롤링을 하는 경우가 많습니다. 이런 크롤링 과정에서 캡차 이미지를 만나는 경우가 종종 있게 됩니다. 캡차 이미지를 만났을 때, 사람이 입력하거나 우회하는 방식이 아니라 정공법으로 인공지능을 통해 컴퓨터가 캡차 이미지의 숫자와 문자를 읽고 해석할 수 있는 프로그램을 만들고 싶었습니다. 따라서 이번 프로젝트 주제로 캡차 이미지 리더기를 만들어 보고자 결정하였습니다.

CAPTCHA_reader를 만든다면 데이터 수집 과정에서 좀 더 편의성이 올라갈 것으로 기대됩니다. 
물론 캡차 이미지를 읽을 수 있는 프로그램은 보안적인 측면에서는 안 좋아 보일 수도 있겠지만 오히려 이런 가능성을 제시하여 보안을 좀 더 강화할 수 있도록 경고하는 의미의 프로젝트가 될 수 있을 것입니다. 

### 데이터
데이터는 kaggle에서 제공하는 CAPTCHA Images 를 활용하였습니다.<br>
https://www.kaggle.com/fournierp/captcha-version-2-images

해당 데이터의 설명입니다.
<br>
Context<br>
This dataset contains CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) images. Built in 1997 as way for users to identify and block bots (in order to prevent spam, DDOS etc.). They have since then been replace by reCAPTCHA because they are breakable using Artificial Intelligence (as I encourage you to do).

Content<br>
The images are 5 letter words that can contain numbers. The images have had noise applied to them (blur and a line). They are 200 x 50 PNGs.

Acknowledgements<br>
The dataset comes from [Wilhelmy, Rodrigo & Rosas, Horacio. (2013). captcha dataset](https://www.researchgate.net/publication/248380891_captcha_dataset).
Thumbnail image from [Accessibility of CAPTCHAs]

Inspiration<br>
This dataset is a perfect opportunity to attempt to make Optical Character Recognition algorithms.
  
### 코드 동작 순서
1. Kaggle 데이터 사용을 위한 초기 세팅 - kaggle & colab 연동 방식 사용.
2. Import Libraries
3. Load Dataset - 데이터 다운받은 폴더로 이동.
4. 이미지 전처리 - Open-CV 활용
5. Label Encoding / OneHotEncoding - 라벨링 되어 있는 이미지를 처리하기 위해 Label Encoding/OneHotEncoding 사용.
6. CNN - 이미지를 학습시키는 것이라 CNN 방식 활용.
7. Fit & Train - batch_size, epochs의 값을 계속 바꾸며 높은 정확도를 가지면서도 효율적으로 동작하는 값 구함. 현재로선 batch_size=64, epochs=100일 때가 최선.
8. Evaluate(평가) - evaluate 함수 결과 출력.
9. 결과 예측 - 이미지 전처리 과정 똑같이 거친 후 모델로 예측하여 실제값과 비교.
10. 모델 저장 - 만들어둔 모델 저장.

**코드에 대한 설명은 .ipynb에 더 적혀있습니다.**

### 프로젝트 결과물
###### 모델의 정확도
![image](https://user-images.githubusercontent.com/49120917/146741511-d7261763-31d6-422f-bed5-e4dcecfedb93.png)
정확도 - 97%

###### 캡차 이미지 예측과 실제 비교
![image](https://user-images.githubusercontent.com/49120917/146743637-6c8922d3-b02f-4240-bc6a-a8fef7ee288f.png)

### 결론
정확도 97%의 CAPTCHA_reader는 위의 프로젝트 결과물과 같이 예측과 결과가 동일하게 나옵니다.<br>

이전에는 정확도 자체가 낮은 값을 가지거나 실질적인 정확도가 떨어진 경우가 많았습니다. 실질적인 정확도의 경우, 정확도는 97%로 동일하더라도 실제 이미지로 예측해봤을 때, 예측값과 실제 결과가 계속 조금씩 다른 경우가 있었습니다. 이미지 전처리 과정에서의 수정이나 모델 훈련 과정에서 수정을 계속 거치면서 실질적인 정확도를 높일 수 있었습니다. 또한, 실질적 정확도 뿐만 아니라 학습 시간도 단축하여 GPU 기준 약 3초 정도 소요되던 것을 1초로 줄일 수 있었습니다. 

이미 기존의 코드가 존재하여 가능한 해당 코드를 활용하였지만, 코드를 계속 공부하고 수정하면서 이전 코드보다 더 정확하고 효율적인 프로젝트 결과물을 만들어 낼 수 있었습니다.<br>앞으로 이 프로젝트를 더 발전하게 된다면 크롤링 프로그램에 모델을 넣어 실제로 크롤링 과정에서 캡차 이미지를 만났을 때, 스스로 처리하도록 동작하는 방식으로 구현하고 싶습니다.<br> 또한, 이번 프로젝트의 데이터는 이미지의 변형이 크게 되지 않고 단순한 blur나 장애물이 추가된 경우로만 제한되어 있었기 때문에 단순한 이미지만 처리 가능할 것으로 예상됩니다. 보다 더 복잡한 캡차 이미지를 읽을 수 있는 프로그램으로 발전해 나가고 싶습니다.


### 참고자료 / Reference
###### 전반적인 코드 참조
https://www.kaggle.com/ritabratanag/captcha-95-accuracy

###### 머신러닝/딥러닝 관련 공부했던 개인 블로그 주소
https://byein.tistory.com/category/ICT%20%EB%A9%98%ED%86%A0%EB%A7%81/%ED%98%BC%EC%9E%90%20%EA%B3%B5%EB%B6%80%ED%95%98%EB%8A%94%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%2B%EB%94%A5%EB%9F%AC%EB%8B%9D

###### kaggle 데이터 colab에서 사용하는 법
https://velog.io/@dunhill741/%EA%B5%AC%EA%B8%80-colab%EC%97%90%EC%84%9C-kaggle-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9D%B4%EC%9A%A9%ED%95%98%EB%8A%94-%EB%B2%95

###### Open-CV
https://hoony-gunputer.tistory.com/entry/opencv-pythonErosion%EA%B3%BC-Dilation

https://diyver.tistory.com/61

https://diyver.tistory.com/67

https://dmcworks-2019-deep-learning.tistory.com/18

###### LabelEncoder, OneHotEncoder
https://mizykk.tistory.com/10

https://3months.tistory.com/207

https://steadiness-193.tistory.com/244

https://hwi-doc.tistory.com/entry/%EB%B2%94%EC%A3%BC%ED%98%95-%EB%B3%80%EC%88%98-%EC%B2%98%EB%A6%AC-%EB%B0%A9%EB%B2%95

https://deepinsight.tistory.com/165

https://khalidpark2029.tistory.com/82

###### test/validation 분리

https://teddylee777.github.io/scikit-learn/train-test-split

###### CNN 
https://buomsoo-kim.github.io/keras/2018/05/02/Easy-deep-learning-with-Keras-8.md/

https://cheris8.github.io/artificial%20intelligence/DL-Keras-Loss-Function/

###### Fitting & Training 
https://sevillabk.github.io/1-batch-epoch/


### License
###### TensorFlow
Copyright 2018 The TensorFlow Authors.
Licensed under the [Apache License, Version 2.0 (the "License")](https://www.apache.org/licenses/LICENSE-2.0);


###### Open-CV
Copyright (C) 2000-2020, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Copyright (C) 2019-2020, Xperience AI, all rights reserved.
Third party copyrights are property of their respective owners.
https://github.com/opencv/opencv/blob/4.4.0/LICENSE

###### Numpy
Copyright (c) 2005-2021, NumPy Developers.
All rights reserved.
https://numpy.org/doc/stable/license.html

###### Pandas
Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
All rights reserved.

Copyright (c) 2011-2021, Open source contributors.

https://github.com/pandas-dev/pandas/blob/master/LICENSE

###### Keras
Keras is licensed under the [MIT license](https://opensource.org/licenses/MIT).

###### scikit-learn
scikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.

###### matplotlib
https://matplotlib.org/stable/users/project/license.html#license-agreement
