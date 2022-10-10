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

Training with Image Augmentation =Pretty Bad
Accuracy on Train is 99%, but on test is 27% and it shows


```
Detecting for class test-tench.jpg model mycnn
--------------------------------
gas pump 0.9362096786499023
church 0.04028371348977089
garbage truck 0.017205338925123215
French horn 0.005383374635130167
golf ball 0.0006115910364314914
--------------------------------
Detecting for class test-church.jpg model mycnn
--------------------------------
parachute 0.675105094909668
tench 0.1729409545660019
golf ball 0.06034903973340988
garbage truck 0.04710496589541435
church 0.030826181173324585
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn
--------------------------------
garbage truck 0.9598315358161926
French horn 0.040168385952711105
chain saw 1.0301283737135236e-07
church 2.5018373150942352e-08
gas pump 6.718596165522506e-12
--------------------------------
Detecting for class test-truck.jpg model mycnn
--------------------------------
cassette player 0.9535076022148132
garbage truck 0.04645875096321106
chain saw 2.4792923795757815e-05
parachute 4.40232224718784e-06
church 2.1299613308656262e-06
--------------------------------
Detecting for class test-dog.jpg model mycnn
--------------------------------
English springer 0.9998047947883606
golf ball 0.00015757889195811003
chain saw 3.541677142493427e-05
tench 9.38796972604905e-07
parachute 5.349552907318866e-07
--------------------------------
Detecting for class test-englishspringer.jpeg model mycnn
--------------------------------
church 0.9971303343772888
parachute 0.0020359137561172247
chain saw 0.0007650338811799884
garbage truck 6.353746721288189e-05
golf ball 4.382040970085654e-06
--------------------------------
```

With Average pooling and Augmentation - Very good

Had the test-dog in training 

```
alex@pop-os:~/coding/cnn_2$ /usr/bin/python3 /home/alex/coding/cnn_2/cnn/test_cnn.py
Detecting for class test-tench.jpg model mycnn
--------------------------------
tench 0.995583713054657
chain saw 0.0029007361736148596
garbage truck 0.0012266312260180712
church 0.00011210612137801945
English springer 0.00010553816537139937
--------------------------------
Detecting for class test-church.jpg model mycnn
--------------------------------
church 0.8739343881607056
French horn 0.09721408039331436
garbage truck 0.028735153377056122
parachute 0.00010893271246459335
gas pump 6.565198873431655e-06
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn
--------------------------------
garbage truck 0.9999979734420776
French horn 2.018884060817072e-06
cassette player 5.524089519148845e-10
church 1.1352511812556809e-11
gas pump 4.400455260594738e-12
--------------------------------
Detecting for class test-truck.jpg model mycnn
--------------------------------
garbage truck 0.9999994039535522
chain saw 3.798146792632906e-07
cassette player 2.4228731376751966e-07
gas pump 1.556056972162878e-08
church 1.0774483527598022e-08
--------------------------------
Detecting for class test-dog.jpg model mycnn
--------------------------------
English springer 1.0
chain saw 8.128528886137241e-28
parachute 1.290010897631937e-29
golf ball 1.664599917784179e-33
church 5.060423742585771e-34
--------------------------------
Detecting for class test-englishspringer.jpeg model mycnn
--------------------------------
English springer 0.9999996423721313
tench 1.9721696276064904e-07
golf ball 1.2497841339609295e-07
French horn 3.878648513477856e-09
garbage truck 3.2409231548458095e-13
```

Training on Imagenette without the testdog image

Accuracy of the network on the 3925 test images: 47.6687898089172 %
Accuracy of the network on the 9469 Train images: 96.68391593621291 %

```
Detecting for class test-tench.jpg model mycnn
--------------------------------
chain saw 0.8944089412689209
tench 0.10334821045398712
French horn 0.0014589522033929825
golf ball 0.0004648714093491435
English springer 0.0003187881375197321
--------------------------------
Detecting for class test-church.jpg model mycnn
--------------------------------
church 0.8240806460380554
garbage truck 0.17589929699897766
French horn 1.1872775758092757e-05
parachute 4.500088834902272e-06
tench 1.3441607507047593e-06
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn
--------------------------------
garbage truck 0.9996967315673828
gas pump 0.00027173792477697134
French horn 1.2280263945285697e-05
tench 1.187692760140635e-05
cassette player 6.54455379844876e-06
--------------------------------
Detecting for class test-truck.jpg model mycnn
--------------------------------
garbage truck 0.8764100670814514
golf ball 0.116241455078125
church 0.005874292459338903
cassette player 0.0012033635284751654
French horn 0.0002699846518225968
--------------------------------
Detecting for class test-dog.jpg model mycnn
--------------------------------
golf ball 0.99992835521698----------------------------------> So the dog image is not really learned
parachute 6.937419675523415e-05
English springer 2.2174074274516897e-06
cassette player 1.3578083156527004e-13
tench 1.1044926773303917e-13
--------------------------------
Detecting for class test-englishspringer.jpeg model mycnn
--------------------------------
English springer 0.9995927214622498
tench 0.0003825757303275168
chain saw 2.389874316577334e-05
French horn 5.999822860758286e-07
golf ball 1.2885385558547569e-07
--------------------------------
```

Training with 1000 images for dogs from imagenette
path = "mycnn_19:31_October062022.pth" 
Accuracy on 3925 test/validation images = 48.12%
Accuracy on 12198 Train images = 97.7 %

```
Detecting for class test-tench.jpg model mycnn
--------------------------------
tench 0.9752317667007446
chain saw 0.022564105689525604
French horn 0.0010246045421808958
golf ball 0.0006363010033965111
English springer 0.0004985574632883072
--------------------------------
Detecting for class test-church.jpg model mycnn
--------------------------------
church 0.8186447620391846
tench 0.08395818620920181
gas pump 0.07923215627670288
garbage truck 0.015643863007426262
golf ball 0.0011756749590858817
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn
--------------------------------
garbage truck 0.9395721554756165
church 0.022244060412049294
English springer 0.017220137640833855
tench 0.012051431462168694
gas pump 0.008629129268229008
--------------------------------
Detecting for class test-truck.jpg model mycnn
--------------------------------
garbage truck 0.8234647512435913
church 0.1747487336397171
chain saw 0.0006748531013727188
parachute 0.00037219308433122933
golf ball 0.0003543358179740608
--------------------------------
Detecting for class test-dog.jpg model mycnn
--------------------------------
English springer 0.9953160285949707 ---------> Good
golf ball 0.004683927167207003
tench 1.2804451943182471e-09
parachute 9.113509946701015e-11
cassette player 3.653466071340539e-11
--------------------------------
Detecting for class test-englishspringer.jpg model mycnn
--------------------------------
chain saw 0.9999861717224121 -----------------> bad and pretty strange
golf ball 8.905326467356645e-06
church 2.9996963348821737e-06
English springer 1.4705417470395332e-06
French horn 4.584421162689978e-07
--------------------------------
Detecting for class test_dogcartoon.jpg model mycnn
--------------------------------
English springer 0.738144040107727       --------------> detected cartoon dog as dog - good
parachute 0.2568896412849426
golf ball 0.0037569578271359205
garbage truck 0.0009252948802895844
cassette player 0.00023350585252046585
--------------------------------
```