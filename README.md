# İleri Tarihli Elektrik Tüketim Verilerinin hava durumu bilgisine göre LSTM modeli ile Tahmini / Forecast of Future Electricity Consumption Data with LSTM model according to weather information

## Proje Özellikleri / Project Features
Proje kapsamında hedefimiz,  elektrik tüketim verilerine paralel olarak eklenmiş hava durumu verileri ile
eğittiğimiz LSTM yapay sinir ağı modeline, belirlenecek bir periyotta elektrik tüketim miktarlarını tahmin ettirmektir.</br>
***Within the scope of the project, our aim is to predict the electricity consumption amounts in a determined period to 
the LSTM artificial neural network model, which we have trained with the weather data added in parallel with the electricity consumption data.***


### Veri Setinin Okunması / Reading the Data Set

Veri setimizde 2473 tane kayıt bulunmaktadır. Her kayıt aralığı üç saate karşılık gelmektedir. Bir günlük verimiz sekiz tane kayıttan oluşmaktadır.
Eğitim kısmı için 2426 tane kayıt kullanılmıştır. Geri kalan kısımın elektrik tüketim bilgilerini silerek sadece hava durumu bilgilerini kullanarak 
kalan kırk sekiz tane kayıt için tahmin üretilmiştir.</br>
***There are 2473 records in our data set. Each recording interval corresponds to three hours. Our one-day data consists of eight records. 
For the training part, 2426 records were used. Forecasts were produced for the remaining forty-eight records using only the weather information, 
deleting the electricity consumption information of the remaining part.***

Önceden hazırlamış olduğumuz veri seti olan “data_1.csv” ve “data_2.csv” isimli dosyamızı tarih-zaman aralıklarını gösteren “Date time” isimli 
sütunları index olarak seçtikten sonra kayıtları parse işlemi yapılmıştır. Sıfır ile beş arasındaki sütunlar “data_1.csv” den alınarak liste haline 
getirilmiştir. Seçilen sütunların özellikleri şu şekildedir.</br>
***After selecting the "data_1.csv" and "data_2.csv" files, which we have prepared before, as an index, the columns named "Date time" showing the date-time 
intervals were selected as an index. Columns between zero and five were taken from “data_1.csv” and turned into a list. The properties of the selected columns are as follows.***

* Birinci sütun olan “Consumption(kWh)” elektrik tüketim değerlerinin tutulduğu kısımdır.</br>
***The first column, “Consumption(kWh)” is the part where electricity consumption values are kept.***
* İki, üç ve dördüncü sütunlar sırasıyla"Maximum Temperature", "Minimum Temperature", "Temperature" olarak isimlendirilerek günün maksimum, 
minimum ve ortalama sıcaklık değerlerinin tutulduğu kısımdır.</br>
***The second, third and fourth columns are named as "Maximum Temperature", "Minimum Temperature", "Temperature", respectively,
It is the part where the minimum and average temperature values are kept.***
* Beşinci sütun olan "Relative Humidity" nem oranı değerlerinin tutulduğu kısımdır.</br>
***The fifth column, "Relative Humidity", is the part where humidity values are kept.***


 ### Veri Setini Ölçekleme ve Örnekleme / Scaling and Sampling Data Set
	
Oluşturacak olduğumuz yapay sinir ağı modeli olan LSTM’ye verileri vermeden önce bazı işlem adımlarından geçmesi gerekmektedir. İlki  daha önceden hazırlamış
olduğumuz üçer saatlik zaman aralıkları boyunca ayarlanmış değerler halinde olmasıdır. İkinci adım olarak ağı eğitecek olduğumuz veri setindeki değerleri sıfır 
ile bir arasında olacak şekilde ölçeklenmesi işlemidir. Bu adımda iki tane ölçeklenmiş veri dizisi oluşturulmuştur. Bunlardan ilki ağı eğitmek için vereceğimiz 
bütün sütun değerleridir. İkinci olarak ise tahmin edilmesini istediğimiz elektrik tüketim değerlerini içeren ilk sütun değerleridir. </br>
***Before giving the data to LSTM, which is the artificial neural network model we will create, it needs to go through some processing steps. 
The first is that it is in the form of adjusted values during the three-hour time intervals that we have prepared before. The second step is to scale the values 
in the data set that we will train the network to be between zero and one. In this step, two scaled data series are created. The first of these is all the column 
values we will give to train the network. Secondly, it is the first column values containing the electricity consumption values that we want to be estimated.***

Ölçekleme işlemini bitirdikten sonraki aşama ise veri seti üzerinde örneklemeler oluşturma adımıdır. Bunun için önceki adımda oluşturduğumuz iki tane ölçeklenmiş 
veri dizilerini girecek olduğumuz “past_step” ve “future_step” kayıt sayısına göre X_train(ilk ölçeklenmiş veri dizisi) ile Y_train(ikinci ölçeklenmiş sütun)  ayarlanarak
matrisler oluşturulmuştur. Bu matrislerde veri seti üzerinde gezinme işlemi yaparak örneklemeler yapılmıştır. Oluşturulan örnekleme sayıları ne kadar iyi düşerse sonuç
olarak alınacak olan tahminlerde o kadar tutarlı olacaktır. </br>
***After completing the scaling process, the next step is to create samples on the data set. For this, matrices are created by setting X_train (first scaled data array) and
Y_train (second scaled column) according to the number of “past_step” and “future_step” records, where we will enter the two scaled data series that we created in the previous
step. Samples were made by navigating on the data set in these matrices. The better the sampling numbers generated, the more consistent the resulting estimates will be.***

### LSTM Tabanlı Sinir Ağının Oluşturulması / Creation of LSTM Based Neural Network
Veri setini sinir ağımıza uygun şekilde ayarlanmıştır. LSTM sinir ağını oluşturmak için hazırladığımız model Keras’ın Sequential modelidir. Sıralı katmanlardan oluşan bir model
yapısı bulunmaktadır. Bir girdi bir ara katman ve bir çıkış katmanı ile model oluşturulmuştur. </br>
***The dataset is tuned to suit our neural network. The model we prepared to create the LSTM neural network is the Sequential model of Keras. There is a model structure consisting of sequential layers. 
The model is constructed with an input layer and an output layer.***

Giriş katmanında yüz on iki tane düğüm yani nöron bulunmaktadır. Bu katmandan ondörtlü paketler halinde ara katmandaki sekiz nörona iletilmektedir. Çıkış katmanında ise bir nöron 
bulunmaktadır. Katmandaki düğüm sayısını arttırmak, model kapasitesini arttırdığı gibi modelin eğitim süresini ve diskte kapladığı alan artırdığı için bu durum istenmemektedir.</br>
***There are one hundred and twelve nodes, or neurons, in the input layer. It is transmitted from this layer in packets of fourteen to eight neurons in the intermediate layer. There is a neuron in the output layer. 
This is undesirable as increasing the number of nodes in the layer increases the model capacity as well as the training time of the model and the space it takes up on the disk.***

Giriş ve ara katmanda kullanmış olduğumuz “return_sequences” parametre değerine “True” değeri verdiğimizde giriş dizisinde gelen her öğe için bir değer üretilirken, “False”
değeri verdiğimizde giriş dizesindeki değerlerin hepsi için bir değer üretmesi anlamına gelmektedir.</br>
***When we give a value of “True” to the “return_sequences” parameter value that we used in the input and middleware, a value is generated for each incoming item in the input array, 
while a value of “False” means that it produces a value for all the values in the input string.***

Ara katman ile çıkış katmanı arasında ağın eğitim için verilmiş olan verileri öğrenirken ezberleme yapmaması için “Dropout” isimli bir katman eklemesi yapılmıştır. </br>
***A layer called "Dropout" has been added between the intermediate layer and the output layer so that the network does not memorize the data given for training while learning.***

Çıkış katmanından elde edilen verilere aktivasyon fonksiyonu uygulanmazsa çıkış sinyali basit bir doğrusal fonksiyon olmaktadır. Doğrusal fonksiyonlar yalnızca tek dereceli polinomlardır.
Aktivasyon fonksiyonu kullanılmayan bir sinir ağı sınırlı öğrenme gücüne sahip bir doğrusal bağlanım (linear regression) gibi davranmaktadır. Sinir ağımızın doğrusal olmayan durumları
da öğrenmesini istediğimiz için aktivasyon fonksiyonu olarak “Doğrusal (Linear) Fonksiyon” kullanılmıştır.</br>
***If the activation function is not applied to the data obtained from the output layer, the output signal becomes a simple linear function. Linear functions are only polynomials of odd degree. 
A neural network without activation function behaves like a linear regression with limited learning power. Nonlinear states of our neural network
"Linear Function" was used as the activation function because we wanted it to learn in***

Modelin derlenmesi “model.compile()” fonksiyonu ile yapılmıştır. Bu fonksiyon optimizer, kayıp olmak üzere iki parametre kullanılmıştır. </br>
***Compilation of the model was done with the "model.compile()" function. This function uses two parameters: optimizer and loss.***

Optimizer, öğrenme oranını kontrol etmektedir. Optimizer olarak “adam” kullanılmaktadır. Adam algoritması, eğitim boyunca öğrenme oranını ayarlamaktadır. Öğrenme oranı, model için optimal
ağırlıkların ne kadar hızlı hesaplandığını belirlemektedir. Daha küçük bir öğrenme oranı daha kesin ve iyi ağırlıklara (belirli bir noktaya kadar) yol açabilir, bu modelin daha iyi öğrenmesi
anlamına gelir ancak ağırlıkların hesaplanması için gereken süre daha uzun olacağından dolayı eğitim süresi uzamaktadır. Kurduğumuz model yapımızda “0.01” olarak ayarlanmıştır. 
Loss(kayıp) fonksiyonları ağın eğitimi esnasında ağın ileri doğru çalışıp ürettiği çıkış değerleri ile gerçek çıkış değerlerini karşılaştırıp, ağın eğitilmesi için geri besleme değeri üreten 
fonksiyonlardır. Modelde kullandığımız kayıp fonksiyonumuz için “mean_squared_error” kullanılmıştır.</br>
***The optimizer controls the learning rate. “Man” is used as the optimizer. The Adam algorithm adjusts the learning rate throughout the training. 
The learning rate is determines how fast the weights are calculated. A smaller learning rate may result in more precise and good weights (up to a certain point), which means the model learns better,
but the training time is longer as the time needed to calculate the weights will be longer. The model we set up is set to “0.01” in our structure. Loss functions are the functions that produce
a feedback value for training the network by comparing the output values produced by the network during the training of the network with the output values produced by the forward. 
“mean_squared_error” is used for our loss function used in the model.***

### Yapay Sinir Ağının(YSA) Ölçeklenmiş Veri Seti ile Eğitilmesi / Training Artificial Neural Network (ANN) with Scaled Dataset

Eğitim için, modelimizdeki “model.fit()” işlevini aşağıdaki yedi parametre ile birlikte kullanılmıştır. </br>
***For training, we used the “model.fit()” function in our model with the following seven parameters.***

* Eğitim verileri (X_train)</br>
***Training data (X_train)***

* Hedef veriler (Y_train)</br>
***Target data (Y_train)***

* Doğrulama seti(validation_data) ile, eğitim sırasında model performansının test edilmesi için kullanılmaktadır.</br>
***With the validation set (validation_data), it is used to test model performance during training.***

* Batch sayısı(batch_size) modelin eğitilmesi aşamasında aynı anda kaç adet verinin işlendiği anlamına gelmektedir. Bizim modelimizde batch size=256 olarak ayarlanmıştır. </br>
***Batch number (batch_size) means how many data are processed at the same time during the training of the model. In our model, batch size=256 is set.***

* Suffle boolean değer alır ve her bir epoch’tan önce verilerin yerlerinin değiştirilmesi için kullanılmaktadır. </br>
***Suffle takes boolean value and is used to swap data before each epoch.***

* Verbose 1 progress bar gibi anlık olarak güncellenen sonuçları göstermektedir. </br>
***Verbose 1 shows results that are updated instantly, such as a progress bar.***

* Epoch sayısı, model eğitilirken verilerin modelden kaç kez geçiş yapacağını belirtmektedir. Modelde epoch değeri elli olarak ayarlanmıştır.</br>
***The epoch number indicates how many times the data will pass through the model while the model is being trained. The epoch value is set to fifty in the model.***

### YSA ile İleri Zamanlı Tahmin Yapma / Forecasting with ANN

X_train dizisinde elektrik tüketimi içermeyen kayıtların sayısı yani “future_step” kadarlık kısım tahmin edilirken değerler “future_step_values_scaled” dizisinde tutulmaktadır. 
Tahmin edilen kayıt sayısı kadar üçer saatlik olacak şekilde tarih-zaman indeksi oluşturulmaktadır.</br>
***While estimating the number of records that do not contain electricity consumption, namely “future_step” in the X_train array, the values are kept in the “future_step_values_scaled” array.
A three-hour date-time index is created as the estimated number of records.***

### Ölçekli Olarak Tahmin Edilen Veriyi Orjinal Haline Getirme / Restoring the Scaled Estimated Data to the Original

Tahmin ettiğimiz değerleri ölçeklenmemiş haline getirmelidir. Bunu yapabilmek için ise “inverse_transform” fonksiyonunu kullanılmıştır. 
Grafikler üzerinde aynı indekslere sahip tarih zamanlarındaki değerleri görebilmek için “TRUE_VALUES”, “TRAIN_VALUES”, “FUTURE_TRUE_VALUES” adında üç tane veri yapısı oluşturulmuştur. </br>
***It should make the values we estimate unscaled. In order to do this, the "inverse_transform" function is used. 
Three data structures named “TRUE_VALUES”, “TRAIN_VALUES”, “FUTURE_TRUE_VALUES” have been created in order to see the values in date times with the same indexes on the graphs.***

### YSA’nın Tahmin Ettiği Değerlerin Grafiksel Olarak Gösterilmesi / Graphical Display of the Estimated Values of the ANN

Elde ettiğimiz  üç farklı veri yapısını kullanarak grafikler oluşturulmuştur. İlk grafik olarak sıcaklık değerlerine bağlı olarak yapılan eğitim ve tahmin değerleri gösterilmiştir. </br>
***Graphs were created using the three different data structures we obtained. Training and prediction values based on temperature values are shown in the first graphic.***

![image](https://github.com/huseyincatalbas/electricity-consumption-forecast-based-on-weather-information/blob/master/LSTM-S%C4%B1cakl%C4%B1k%20Aras%C4%B1ndaki%20%C4%B0li%C5%9Fki.jpeg)
![image](https://github.com/huseyincatalbas/electricity-consumption-forecast-based-on-weather-information/blob/master/LSTM-Nem%20Oran%C4%B1%20Aras%C4%B1ndaki%20%C4%B0li%C5%9Fki.jpeg)
![image](https://github.com/huseyincatalbas/electricity-consumption-forecast-based-on-weather-information/blob/master/LSTM-Tahmin%20Edilen%20Elektrik%20T%C3%BCketimi%20ile%20Kullan%C4%B1lan%20T%C3%BCketimin%20Aras%C4%B1ndaki%20%C4%B0li%C5%9Fki.jpeg)
