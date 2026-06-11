# -*- coding: utf-8 -*-
"""Uc veri setini 3000'er kayda cikarir.
Kalite odakli: anlamsal uyumlu gruplar, genis havuz, dedup + tam BIO dogrulama."""
import json, re, os, random
random.seed(2024)
HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set(list(".,;:!?\"'()…"))

def tokenize(word):
    toks=[]; i=0
    while i<len(word) and word[i] in PUNCT: toks.append(word[i]); i+=1
    word=word[i:]; tail=[]
    while word and word[-1] in PUNCT: tail.insert(0,word[-1]); word=word[:-1]
    if word: toks.append(word)
    toks.extend(tail); return toks

def build(sentence,label):
    tokens,labels=[],[]
    for part in re.split(r"(«[^»]*»)",sentence):
        if not part: continue
        if part.startswith("«") and part.endswith("»"):
            first=True
            for w in part[1:-1].strip().split():
                for t in tokenize(w):
                    if t in PUNCT: tokens.append(t); labels.append("O"); first=True
                    else: tokens.append(t); labels.append(("B-" if first else "I-")+label); first=False
        else:
            for w in part.split():
                for t in tokenize(w): tokens.append(t); labels.append("O")
    return {"tokens":tokens,"labels":labels}

def validate(ex,s):
    assert len(ex["tokens"])==len(ex["labels"]),"len: "+s
    assert any(l!="O" for l in ex["labels"]),"etiket yok: "+s
    assert ex["tokens"][0][0].isupper() or not ex["tokens"][0][0].isalpha(), "buyuk harf degil: "+s
    prev="O"
    for l in ex["labels"]:
        assert not (l.startswith("I-") and prev not in ("B-"+l[2:],"I-"+l[2:])),"BIO: "+s
        prev=l

def fill_file(filename,label,candidates,target):
    path=os.path.join(HERE,filename)
    data=json.load(open(path,encoding="utf-8"))
    existing={" ".join(x["tokens"]) for x in data}
    random.shuffle(candidates)
    added=skipped=0
    for s in candidates:
        if len(data)>=target: break
        s=re.sub(r"\s+"," ",s).strip()
        ex=build(s,label); validate(ex,s)
        key=" ".join(ex["tokens"])
        if key in existing: skipped+=1; continue
        existing.add(key); data.append(ex); added+=1
    json.dump(data,open(path,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    warn="" if len(data)>=target else f"  [EKSIK {target-len(data)}]"
    print(f"{filename}: +{added} eklendi, {skipped} kopya, toplam {len(data)}{warn}")

# ======================================================= ABARTI
LEAD_A = ["Bugün","Resmen","Adeta","Cidden","Vallahi","Gerçekten","Az önce","Demin",
          "Bu aralar","Son günlerde","Şu an","Az kalsın","Bir anda","Sabahtan beri",
          "Bütün gün","O gün","İnan ki","Dün","Akşama kadar","Şu sıralar","Bütün gece","Hâlâ"]
TRAIL_A = ["", "resmen.", "valla.", "gerçekten.", "cidden.", "abartmıyorum.",
           "yemin ederim.", "inan.", "diyebilirim.", "desem yeridir."]
# YENI cekirdekler (pre, span, post); span = abartilan kisim
CORE_A = [
 # para / is
 ("bu ay","faturalar","kemerleri delik deşik etti"), ("bu zamlarla","cep","sızlıyor"),
 ("market alışverişi","servet","götürüyor"), ("bu borç","beni","mezara sokacak"),
 ("kira","uçtu da","gitti"), ("maaş","göz açıp kapayana kadar","eridi"),
 ("bu işte","dünyanın parasını","döktük"), ("masraflar","baş döndürücü","seviyede"),
 # teknoloji / okul
 ("bu ödev","beynimi","eritti"), ("sınav konuları","kafamı","şişirdi"),
 ("bu ders","bir asır","sürdü"), ("telefonun şarjı","göz açıp kapayana kadar","bitti"),
 ("bu rapor","gece yarılarıma","mal oldu"), ("notları ezberlemek","ömrümü","yedi"),
 # trafik / yol
 ("trafikte","saçım","ağardı"), ("bu yolda","tekerlekler","eridi"),
 ("kuyrukta","emekli","oldum"), ("otobüs","kaplumbağadan yavaş","gidiyordu"),
 ("bu yokuş","göğe dayalı","gibiydi"), ("yol","bir türlü bitmek","bilmedi"),
 # hava
 ("bu sıcakta","kaldırımlar","cayır cayır yandı"), ("güneş","tepemizde","cehennem oldu"),
 ("bu soğukta","kanım","damarımda dondu"), ("rüzgâr","çatıları","uçurdu"),
 ("yağmur","gökten sel gibi","boşandı"), ("kar","kapıları","yuttu"),
 ("bu sisle","elimi","göremedim"), ("dolu","cevizden iri","yağdı"),
 # beden / yorgunluk
 ("bu koşudan sonra","ciğerlerim","ağzıma geldi"), ("ayaklarım","kütük","gibi şişti"),
 ("baş ağrısından","kafam","ikiye bölünecek"), ("bu uykusuzlukla","gözüm","çukura kaçtı"),
 ("bacaklarım","kurşun kadar","ağırlaştı"), ("sırtım","yük altında","çöktü"),
 # duygular
 ("bu haberle","sevincimden","yerimde duramadım"), ("öyle bir korktum ki","kalbim","yerinden fırladı"),
 ("özlemekten","içim","yandı bitti"), ("bu kavgada","öfkeden","gözüm döndü"),
 ("utancımdan","yerin dibine","geçtim"), ("heyecandan","ellerim","buz kesti"),
 ("o sahnede","gözyaşlarım","sel oldu"), ("bu sevinçle","ayaklarım","yerden kesildi"),
 # yemek / aclik
 ("bu sofrada","bir ordu","doyardı"), ("açlıktan","mideme kurtlar","düştü"),
 ("bu yemeğe","bayıla bayıla","yedim"), ("o kadar yedim ki","patlamak","üzereyim"),
 ("bu tatlı","damağımda","bayram ettirdi"), ("susuzluktan","dudaklarım","çatladı"),
 # kalabalik / ses
 ("meydan","insan kaynıyor","mahşer gibiydi"), ("bu gürültüden","kulaklarım","sağır oldu"),
 ("konserde","yer yerinden","oynadı"), ("alkıştan","tavan","çöktü"),
 ("bu kalabalıkta","toplu iğne","yere düşmezdi"), ("çocukların sesi","beynimi","deldi"),
 # zaman / mesafe / miktar
 ("onu","yüzyıllardır","görmedim"), ("bu sıra","kilometrelerce","uzadı"),
 ("ödevler","dağ gibi","birikti"), ("sözünü","binlerce kez","tekrarladım"),
 ("bu kitap","tuğla kadar","kalındı"), ("valiz","fil kadar","ağırdı"),
 ("bekleyeli","kırk yıl","oldu"), ("anlatacaklarım","ciltlere","sığmaz"),
 # genel kalip abartilar
 ("","kıyamet","koptu"), ("","ortalık","savaş alanına döndü"),
 ("","dünya başıma","yıkıldı"), ("","yer yerinden","oynadı"),
 ("o anda","zaman","durdu"), ("bu haber","bomba gibi","düştü"),
 ("onun zenginliği","denizdeki kum kadar",""), ("onun sabrı","taşı bile","utandırır"),
 ("onun cömertliği","sınır","tanımaz"), ("onun yalanları","kitap","olur"),
 ("onun zekâsı","şeytana pabucu ters","giydirir"), ("onun öfkesi","yanardağ gibi","patladı"),
 ("bu acı","yürekleri","dağladı"), ("bu güzellik","akılları","durdurdu"),
 ("onun gülüşü","ortalığı","aydınlattı"), ("bu söz","içimi","paramparça etti"),
 ("bu film","beni","koltuğa mıhladı"), ("bu manzara","nefesimi","kesti"),
 ("o haber","aklımı","başımdan aldı"), ("bu başarı","göğsümü","kabarttı"),
]

def gen_abarti():
    out=[]
    for pre,span,post in CORE_A:
        leads=random.sample(LEAD_A, 9)
        for lead in leads:
            body = lead + " " + ((pre+" ") if pre else "") + "«"+span+"»" + ((" "+post) if post else "")
            for tr in random.sample(TRAIL_A, 3):
                out.append(body+(" "+tr if tr else "."))
    return out

# ======================================================= MECAZ
TGT=["içimi","yüreğimi","kalbimi","ruhumu","zihnimi","benliğimi","aklımı","gönlümü"]
NEG_ABS=["Korku","Kaygı","Öfke","Hüzün","Endişe","Acı","Keder","Nefret","Yalnızlık",
         "Pişmanlık","Şüphe","Üzüntü","Dehşet","Kuşku","Telaş","Panik","Çaresizlik",
         "Kıskançlık","Hasret","Gam","Tasa","Kuruntu","Öfke","Korku","Vicdan azabı"]
NEG_V=["kapladı","sardı","kemirdi","kuşattı","boğdu","bürüdü","esir aldı","yiyip bitirdi",
       "sarıp sarmaladı","içine aldı","gölgeledi","felç etti","zincirledi","kavurdu"]
POS_ABS=["Sevgi","Umut","Sevinç","Mutluluk","Coşku","Neşe","Huzur","Şefkat","Heyecan","Gurur"]
POS_V=["sardı","kapladı","ısıttı","doldurdu","kuşattı","bürüdü","sarıp sarmaladı","aydınlattı"]
LOC=["içimde","yüreğimde","kalbimde","zihnimde","ruhumda","gönlümde","benliğimde"]
GROW_ABS=["Umut","Sevgi","Sevinç","Coşku","Özlem","Hayranlık","İnanç","Cesaret","Merhamet",
          "Şefkat","Tutku","Heves","Sevda","Saygı"]
GROW_V=["filizlendi","yeşerdi","kök saldı","büyüdü","çiçek açtı","boy verdi","tomurcuklandı",
        "yeniden doğdu","kanat çırptı","serpildi"]
NEGGROW_ABS=["Korku","Kin","Şüphe","Nefret","Pişmanlık","Öfke","Kaygı","Kuşku","Hınç","Garez"]
NEGGROW_V=["depreşti","kök saldı","büyüdü","kabardı","uyandı","yeniden alevlendi","içine yerleşti",
           "tırmandı","derinleşti"]
OBJ_SUBJ=["Bu söz","Bu haber","Bu olay","Bu karar","Bu fikir","Söyledikleri","Onun sözleri",
          "Bu cümle","O bakış","Bu anı","Bu mektup","O an"]
OBJ_TGT=["yüreğime","kalbime","içime","ruhuma","gönlüme","zihnime"]
OBJ_V=["dokundu","işledi","kazındı","saplandı","battı","ok gibi saplandı","su serpti",
       "taht kurdu","iz bıraktı","mührünü vurdu","derin yara açtı"]
# Dogal sifat mecazlari (curated)
ADJ_M = [
 "Onun «çelik» iradesi karşısında kimse duramadı.",
 "Onun «buz» gibi tavrı içimi ürpertti.",
 "Onun «altın» kalbi her derde derman oldu.",
 "Onun «zehir» dili dostluğu zehirledi.",
 "Onun «bal» gibi sözleri gönlümü okşadı.",
 "Onun «kara» talihi yıllarca peşini bırakmadı.",
 "Onun «sıcak» yüreği herkese kapı açtı.",
 "Onun «soğuk» bakışı araya buz koydu.",
 "Onun «keskin» zekâsı her düğümü çözdü.",
 "Onun «tatlı» dili taşı bile yumuşatır.",
 "Onun «kadife» sesi içime su gibi aktı.",
 "Onun «kömür» gözleri ruhuma işledi.",
 "Onun «pamuk» elleri yaramı sardı.",
 "Onun «demir» yumruğu masaya indi.",
 "Onun «cam» kalbi en küçük sözde kırıldı.",
 "Onun «alev» bakışları içimi tutuşturdu.",
 "Onun «buruk» gülüşü beni hüzünlendirdi.",
 "Onun «engin» gönlü herkesi içine aldı.",
 "Onun «yıldız» misali zekâsı parıldadı.",
 "Onun «mahzun» bakışı yağmur bulutu gibiydi.",
 "Onun «sert» sözleri içimi kanattı.",
 "Onun «pırıltılı» fikirleri sohbete ışık tuttu.",
 "Onun «taş» kalbini hiçbir gözyaşı eritemedi.",
 "Onun «ipek» gibi nazik tavrı herkesi etkiledi.",
 "Onun «zümrüt» bakışı büyüledi hepimizi.",
 "Onun «sıcacık» kucağı bütün korkularımı dağıttı.",
]
# Doga / hayat kisilestirme (curated)
LIFE_M = [
 "Hayat ona sonunda «yüzünü döndü».",
 "Hayat bize o yıl «sırtını döndü».",
 "Hayat ona kapılarını «ardına kadar açtı».",
 "Hayat bana yine «oyun oynadı».",
 "Zaman bütün yaraları yavaşça «sardı».",
 "Zaman acıların üstüne «toz kondurdu».",
 "Yıllar onun yüzüne «derin izler» bıraktı.",
 "Yıllar saçlarına usulca «kar» yağdırdı.",
 "Talih sonunda yüzüne «güldü».",
 "Kader onun «yakasını bırakmadı».",
 "Gece bütün sokakları «yuttu».",
 "Şafak karanlığı «dağıttı».",
 "Karanlık vadiyi «kucağına aldı».",
 "Deniz o gece öfkeyle «kükredi».",
 "Sis tepeleri yavaşça «sardı».",
 "Güneş bulutların ardına «saklandı».",
 "Bahar uyuyan doğayı «uyandırdı».",
 "Kış bütün bahçeyi «uykuya yatırdı».",
 "Rüzgâr kuru yaprakları «savurup götürdü».",
 "Şehir beni ışıklarıyla «içine çekti».",
 "Yağmur pencereye usulca «dokundu».",
 "Sabah güneşi odayı «altına boğdu».",
 "Ay ışığı denize «gümüş döktü».",
 "Doğa baharla birlikte «cana geldi».",
]

def gen_mecaz():
    out=[]
    for a in NEG_ABS:
        for t in random.sample(TGT,5):
            for v in random.sample(NEG_V,6):
                out.append(f"{a} {t} «{v}».")
    for a in POS_ABS:
        for t in random.sample(TGT,5):
            for v in POS_V:
                out.append(f"{a} {t} «{v}».")
    for a in GROW_ABS:
        for loc in random.sample(LOC,5):
            for v in random.sample(GROW_V,6):
                out.append(f"{a} {loc} «{v}».")
    for a in NEGGROW_ABS:
        for loc in random.sample(LOC,5):
            for v in NEGGROW_V:
                out.append(f"{a} {loc} «{v}».")
    for s in OBJ_SUBJ:
        for t in random.sample(OBJ_TGT,4):
            for v in random.sample(OBJ_V,7):
                out.append(f"{s} {t} «{v}».")
    out += ADJ_M*4 + LIFE_M*4
    return out

# ======================================================= DEYIM
LEAD_D = ["Bu haberle","O gün","Sonunda","Nihayet","Bir anda","Yine","Resmen","Anlaşılan",
          "Duyunca","Görünce","Az önce","Bu olayla","O günden beri","Nedense","Birden",
          "Çok geçmeden","Beklenmedik şekilde","Maalesef","Neyse ki","Az kalsın","Doğrusu",
          "Gördüğüm kadarıyla","Söylenenlere göre","Her zamanki gibi"]
SUBJ_D = ["Adam","Kadın","Çocuk","Patron","Komşu","Arkadaşım","Müdür","Öğretmen",
          "Yaşlı adam","Genç kız","Babam","Annem","Amcam","Halam"]
CONN_D = ["","duyunca","görünce","öğrenince","sonunda","nihayet","bir anda","yine",
          "nedense","birden","o gün","sözünü duyunca"]
IDIOMS = [
 "küplere bindi","etekleri zil çaldı","ağzı kulaklarına vardı","burnundan kıl aldırmadı",
 "pabucu dama atıldı","kafayı yedi","çileden çıktı","tepesi attı","kanı beynine sıçradı",
 "ağzı açık kaldı","suratı asıldı","keyfi kaçtı","içi içine sığmadı","eli ayağı tutuldu",
 "dili tutuldu","yüreği ağzına geldi","ödü koptu","aklı başından gitti","etekleri tutuştu",
 "foyası ortaya çıktı","pot kırdı","çuvalladı","yelkenleri suya indirdi","pes etti",
 "kolları sıvadı","ağzından baklayı çıkardı","defteri dürdü","ortalığı velveleye verdi",
 "tüyleri diken diken oldu","saçını başını yoldu","gözünü kırpmadı","hapı yuttu",
 "yan gelip yattı","kulak kabarttı","baltayı taşa vurdu","ipe un serdi","pireyi deve yaptı",
 "burnu büyüdü","ağzı sulandı","gözü doymadı","eli boş döndü","kanı kaynadı",
 "içine kurt düştü","içine su serpildi","gönlü ferahladı","gözleri doldu","gözleri parladı",
 "dünyası karardı","dünyası başına yıkıldı","dört gözle bekledi","can attı","canına tak etti",
 "sabrı taştı","sabrı tükendi","kendini kaybetti","kendine geldi","aklı karıştı",
 "kafası karıştı","gözü korktu","eli ayağı buz kesti","dizlerinin bağı çözüldü",
 "beti benzi attı","rengi soldu","yüzü kızardı","yüzü güldü","içi rahatladı",
 "içi parçalandı","yüreği parçalandı","bağrı yandı","ciğeri yandı","hüngür hüngür ağladı",
 "katıla katıla güldü","gülmekten kırıldı","ağzının payını aldı","burnu sürtüldü",
 "kuyruğu titretti","geri adım attı","pabuç bırakmadı","omuz silkti","surat astı",
 "dudak büktü","burun kıvırdı","telaşa kapıldı","paniğe kapıldı","soğukkanlılığını korudu",
 "kılını kıpırdatmadı","istifini bozmadı","oralı olmadı","kulak kesildi","ağırdan aldı",
 "ayağını sürüdü","kös kös oturdu","sevinçten havalara uçtu","mutluluktan uçtu",
 "hayal kırıklığına uğradı","suya sabuna dokunmadı","gözden düştü","göze girdi",
 "yüreğine su serpildi","içi içini yedi","gözü arkada kaldı","yüzü yere baktı",
 "ağzı bir karış açık kaldı","aklı yattı","içi geçti","gözleri faltaşı gibi açıldı",
 "burnu Kaf dağına çıktı","ağzına bir şey almadı","kendini paraladı","dört döndü",
 "gözü kaldı","içi titredi","yüreği hopladı","kafa patlattı","ağzı laf yaptı",
 "etekleri zil çaldı","ipleri eline aldı","çark etti","diş bilemeye başladı",
 "gözünü budaktan sakınmadı","kulak ardı etti","omuz omuza durdu","el ele verdi",
 "bel bağladı","göz koydu","akıntıya kürek çekti","pabucu büyük geldi","tası tarağı topladı",
 "kafasını kuma gömdü","gözden ırak tuttu","kulağına küpe etti","dilinin altında bir şey kaldı",
 "yüreği yağ bağladı","içi içine sığmaz oldu","gözleri kan çanağına döndü","yüzü asıldı",
 "kaşları çatıldı","keyfi yerine geldi","morali bozuldu","morali düzeldi","neşesi kaçtı",
 "öfkesi burnunda dolaştı","sabrı kalmadı","gözü açık gitti","içi yandı","bağrına taş bastı",
]

def gen_deyim():
    out=[]
    for idm in IDIOMS:
        for lead in random.sample(LEAD_D, 12):
            out.append(f"{lead} «{idm}».")
        for subj in random.sample(SUBJ_D, 6):
            c=random.choice(CONN_D)
            out.append((f"{subj} {c} «{idm}»." if c else f"{subj} «{idm}»."))
    return out

if __name__=="__main__":
    fill_file("dataset_abarti.json","ABARTI",gen_abarti(),target=3000)
    fill_file("dataset_mecaz.json","MECAZ",gen_mecaz(),target=3000)
    fill_file("dataset_deyim.json","DEYIM",gen_deyim(),target=3000)
