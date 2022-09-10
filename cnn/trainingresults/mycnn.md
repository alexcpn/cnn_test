With Dropout in CNN and FC
mycnn commit - 8bab8d747c505649445842705ea735c5ef9a2314

Everything is bad
```
Detecting for class test-tench.jpg model mycnn
--------------------------------
cassette player 0.10442381352186203
garbage truck 0.10166455805301666
tench 0.10157012939453125
French horn 0.1013716384768486
parachute 0.1008942723274231
--------------------------------
Detecting for class test-church.jpg model mycnn
--------------------------------
cassette player 0.10442381352186203
garbage truck 0.10166455805301666
tench 0.10157012939453125
French horn 0.1013716384768486
parachute 0.1008942723274231
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn
--------------------------------
cassette player 0.10442381352186203
garbage truck 0.10166455805301666
tench 0.10157012939453125
French horn 0.1013716384768486
parachute 0.1008942723274231
--------------------------------
Detecting for class test-truck.jpg model mycnn
--------------------------------
cassette player 0.10442381352186203
garbage truck 0.10166455805301666
tench 0.10157012939453125
French horn 0.1013716384768486
parachute 0.1008942723274231
--------------------------------
Detecting for class test-dog.jpg model mycnn
--------------------------------
cassette player 0.10442381352186203
garbage truck 0.10166455805301666
tench 0.10157012939453125
French horn 0.1013716384768486
parachute 0.1008942723274231
--------------------------------
Detecting for class test-englishspringer.jpeg model mycnn
--------------------------------
cassette player 0.10442381352186203
garbage truck 0.10166455805301666
tench 0.10157012939453125
French horn 0.1013716384768486
parachute 0.1008942723274231
--------------------------------
```

With Dropout in FC only
```
No major change
```

With extra layers
```
alex@pop-os:~/coding/cnn_2$ python3 cnn/test_cnn.py 
Detecting for class test-tench.jpg model mycnn
--------------------------------
chain saw 0.9983841180801392
French horn 0.0015956725692376494
cassette player 1.9969051209045574e-05
gas pump 1.2364490942218254e-07
garbage truck 9.263445832630168e-08

Detecting for class test-truck.jpg model mycnn
--------------------------------
tench 1.0

Detecting for class test-dog.jpg model mycnn
--------------------------------
golf ball 0.9002022743225098

--------------------------------
Detecting for class test-englishspringer.jpeg model mycnn
--------------------------------
English springer 0.9999775886535645
```
With BathNormalization - slightly better

```
Detecting for class test-tench.jpg model mycnn
--------------------------------
tench 0.84977126121521
gas pump 0.12847349047660828
chain saw 0.02159758470952511
garbage truck 0.00010672565258573741
golf ball 3.1292354833567515e-05
--------------------------------
Detecting for class test-church.jpg model mycnn
--------------------------------
church 0.9752296209335327
garbage truck 0.02371390536427498
French horn 0.0010499705094844103
gas pump 6.3401889747183304e-06
parachute 1.1721299131295382e-07
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn
--------------------------------
garbage truck 0.9607970714569092
gas pump 0.02324913814663887
church 0.015872078016400337
chain saw 2.7475522074382752e-05
cassette player 2.4248867703136057e-05
--------------------------------
Detecting for class test-truck.jpg model mycnn
--------------------------------
cassette player 0.534131646156311
garbage truck 0.46480172872543335     --> not that good
parachute 0.0010520177893340588
tench 5.157675786904292e-06
golf ball 4.8360516302636825e-06
--------------------------------
Detecting for class test-dog.jpg model mycnn
--------------------------------
English springer 0.9237158298492432  ----->Good
tench 0.03902341052889824
golf ball 0.03696732968091965
chain saw 0.00022663446725346148
gas pump 6.548382225446403e-05
--------------------------------
Detecting for class test-englishspringer.jpeg model mycnn
--------------------------------
chain saw 0.8347074389457703 ----> Why
church 0.16420884430408478
French horn 0.0006901451270096004
golf ball 0.000250943994615227
tench 7.570091838715598e-05
--------------------------------
```

Training with Image Augmentation

```
Detecting for class test-tench.jpg model mycnn
--------------------------------
tench 0.9014003872871399
golf ball 0.031259145587682724
chain saw 0.023627735674381256
French horn 0.014319909736514091
English springer 0.009539403021335602
--------------------------------
Detecting for class test-church.jpg model mycnn
--------------------------------
church 0.5142383575439453
French horn 0.4114319980144501
gas pump 0.04233633354306221
cassette player 0.0161536056548357
chain saw 0.008743335492908955
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn
--------------------------------
church 0.42769762873649597 --> Bad
garbage truck 0.401841938495636
gas pump 0.0972723513841629
French horn 0.03204435110092163
golf ball 0.011129915714263916
--------------------------------
Detecting for class test-truck.jpg model mycnn
--------------------------------
garbage truck 0.8944478631019592
chain saw 0.0458856076002121
parachute 0.01629267819225788
French horn 0.01472572423517704
cassette player 0.007777288090437651
--------------------------------
Detecting for class test-dog.jpg model mycnn
--------------------------------
tench 0.47618719935417175 -->bad
golf ball 0.46452292799949646
English springer 0.03245774284005165
parachute 0.017508018761873245
cassette player 0.003098340006545186
--------------------------------
Detecting for class test-englishspringer.jpeg model mycnn
--------------------------------
English springer 0.8400651216506958
golf ball 0.06779954582452774
tench 0.04761919006705284
French horn 0.020101355388760567
parachute 0.01695236936211586
--------------------------------
```
