
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)
# day_diff - Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes - Değerlendirmenin faydalı bulunma sayısı
# total_vote - Değerlendirmeye verilen oy sayısı


import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################

df = pd.read_csv("RatingProductSortingReviewsinAmazon-221119-111357")
df.head()
df["overall"].mean()
df["overall"].describe()
df["overall"].value_counts()
df["overall"].hist()
plt.show()

0.99/4.58

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################

# day_diff: yorum sonrası ne kadar gün geçmiş
df['reviewTime'] = pd.to_datetime(df['reviewTime'])
current_date = pd.to_datetime('2014-12-08 00:00:00')
df["day_diff1"] = (current_date - df['reviewTime']).dt.days
df["day_diff1"].describe()


# zaman bazlı ortalama ağırlıkların belirlenmesi

q1 = df["day_diff"].quantile(0.25) # 281
q2 = df["day_diff"].quantile(0.50) # 431
q3 = df["day_diff"].quantile(0.75) # 601

# a,b,c değerlerine göre ağırlıklı puanı hesaplayınız.

weighted_average=df.loc[df["day_diff"]<= q1, "overall"].mean()*50/100+\
                 df.loc[(df["day_diff"]> q1) & (df["day_diff"]<= q2), "overall"].mean()*25/100+\
                 df.loc[(df["day_diff"]> q2) & (df["day_diff"]<= q3), "overall"].mean()*15/100+\
                 df.loc[df["day_diff"]> q3, "overall"].mean()*10/100

# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
df.loc[df["day_diff"]<= q1, "overall"].mean() # 4.70
df.loc[(df["day_diff"]> q1) & (df["day_diff"]<= q2) , "overall"].mean() # 4.63
df.loc[(df["day_diff"]> q2) & (df["day_diff"]<= q3) , "overall"].mean() # 4.57
df.loc[df["day_diff"]> q3, "overall"].mean() # 4.44

### 2.yöntem fonksiyonlaştırılmış hali

def time_based_weighted_average(dataframe, w1=50, w2=25, w3=15, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df["helpful"].head()

# 1.yol
df["helpful"]=df["helpful"].str.strip('[ ]')
df["helpful_yes"]=df["helpful"].apply(lambda x:x.split(", ")[0]).astype(int)
df["total_vote"]=df["helpful"].apply(lambda x:x.split(", ")[1]).astype(int)
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(50)

# 2.yol
df["helpful_yes"]=df[["helpful"]].applymap(lambda x:x.split(", ")[0].strip('[')).astype(int)
df["total_vote"]=df[["helpful"]].applymap(lambda x:x.split(", ")[1].strip(']')).astype(int)

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

df.head()
# score_pos_neg_diff

def score_up_down_diff(up, down):
    return up - down

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# 2. yöntem
df["score_pos_neg_diff"]=df["helpful_yes"]- df["helpful_no"]


# score_average_rating

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)



# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"].describe([0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.77,0.95,0.97,0.98,0.99,1])
df["wilson_lower_bound"].value_counts()


##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)


### EK

def calculate_bayesian_rating_products(rating_counts,confidence_level=0.95):
    if sum(rating_counts)==0:
        return 0
    # Calculate the expected expected value of the rating distribution
    num_scores=len(rating_counts)
    z=st.norm.ppf(1-(1-confidence_level)/2)
    total_ratings=sum(rating_counts)
    expected_value=0.0
    expected_value_squared=0.0
    for score,count in enumerate(rating_counts):
        probability=(count+1)/(total_ratings+num_scores)
        expected_value += (score + 1) * probability
        expected_value_squared += (score + 1) * (score + 1) * probability
    # Calculate the variance of the rating distribution
    variance=(expected_value_squared-expected_value **2)/(total_ratings+num_scores+1)
    # Calculate the Bayesian avg score
    bayesian_average=expected_value-z*math.sqrt(variance)
    return bayesian_average


