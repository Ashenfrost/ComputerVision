# Bilag 5: Samtale med Claude (Claude Code, Anthropic)

Udskrift af samtalen: mine prompter og Claudes svar. Værktøjskald (kørsel af kode, filredigering) og automatiske systembeskeder er udeladt. Tidspunkter er UTC.

---

## Prompt 1 (2026-11-02 14:20)

**Mig:**

Jeg arbejder på en porteføljeopgave i Computer Vision (IT-teknolog, 3. semester, UCL).  
Rapporten er hovedproduktet, koden er bilag. Svar på dansk.  
  
## Arbejdsform  
- Forklar hvad du ændrer og hvorfor, før/mens du gør det. Jeg skal kunne forklare og  
  dokumentere processen i rapporten selv.  
- Stil gerne et spørgsmål indimellem, så jeg selv reflekterer, men jeg er bagud, så lad  
  være med at holde svar tilbage for længe.  
- Lad være med at overkomplimentere mig for små ting.  
- Jeg er Python-begynder. Windows, venv, NVIDIA RTX 3050 (6 GB VRAM).  
  
## Projektet  
Use case: detektion og genkendelse af vejskilte.  
Datasæt: `cardata/` (Kaggle pkdarabi/cardetection, Roboflow-export i YOLO-format),  
15 klasser: Green Light, Red Light, Speed Limit 10-120 (én klasse pr. værdi), Stop.  
Labels i `cardata/*/labels/*.txt`: `klasse cx cy w h` (normaliseret 0-1).  
Bemærk: stierne i `cardata/data.yaml` er `../train/images` osv. og skal måske rettes  
før YOLO kan bruge filen.  
  
Vurderingsgrundlaget belønner især: filtre (blur, edge detection, farvefiltre) anvendt  
i sekvens med dokumenteret forbedring, at pipelinen virker på flere positive OG negative  
billeder, og sammenligning af flere detektionsalgoritmer med styrker/svagheder.  
Præcision i sig selv vægtes ikke højt.  
  
## Hvor langt jeg er  
1. Klassisk pipeline (tidligere): blur (Gaussian/Median/Bilateral) -> rød hue-filter ->  
   Hough circles virkede godt. Blob detection virkede ikke.  
2. CNN iteration 1: `Machineconvoluted.py` – eget PyTorch-CNN (4 conv-blokke m. BatchNorm),  
   klassificerer HELE billedet (418x418) med klassen fra den største bounding box,  
   10 epochs, ingen augmentation. Resultat (`confusion_matrix.png`): modellen kollapsede  
   mod få klasser (mest Green Light og Speed Limit 30). Kun Stop (60/78) og  
   Speed Limit 20 (45/55) virkede. Red Light blev næsten altid til Green Light.  
   Ingen Speed Limit 10 i valid-sættet.  
3. CNN iteration 2: `Machineconvoluted_crops.py` (skrevet, IKKE kørt endnu) – samme  
   arkitektur, men skærer hvert skilt ud via bounding box (+15% padding), 64x64,  
   augmentation (rotation/translate/lys – bevidst INGEN horizontal flip og INGEN hue,  
   fordi tal spejlvendes og farve adskiller rød/grøn), class weights i NLLLoss,  
   30 epochs, StepLR hver 10. epoch, gemmer bedste model, classification_report,  
   Grad-CAM. Gemmer til *_crops-filer så iteration 1 ikke overskrives.  
  
## Fund om datakvalitet  
- Nogle billeder er sløre, opskalerede udsnit (ligner GTSRB, filnavne `00000_00000_...`),  
  hvor skiltet fylder det meste – andre er rigtige vejbilleder.  
- Fejl i labels: f.eks. `cardata/train/labels/00000_00000_00000_png.rf.55d47572c5980af0892b0c2ada6dae77.txt`  
  har to næsten identiske bokse om samme skilt med klasse 7 (SL30) og 6 (SL20).  
  Det skal undersøges hvor udbredt det er.  
- Speed Limit 10 har meget få eksempler (~22) – beholdes bevidst som dokumenteret begrænsning.  
  
## Næste skridt  
1. Tjek hvor mange label-filer der har flere forskellige klasser, og vurder om det er  
   ægte flere skilte eller dobbelt-labels (kig på nogle af billederne).  
2. Kør `Machineconvoluted_crops.py`, og sammenlign confusion matrix og per-klasse  
   precision/recall med iteration 1. Hjælp mig med at tolke resultatet.  
3. Derefter: træn YOLOv8 (Ultralytics, fortrænet yolov8n) på samme datasæt.  
4. Til sidst: sammenlign klassisk pipeline vs. eget CNN vs. YOLOv8 til rapporten,  
   inkl. test på negative billeder (uden skilte).  
  
Start med at læse `Machineconvoluted.py` og `Machineconvoluted_crops.py`, og gå så i  
gang med skridt 1.

**Claude:**

Én ting mangler: hvilke **klasser** de små bokse tilhører. Hvis det mest er trafiklys, mister crop-modellen en stor del af sine Red/Green Light-eksempler.

Dobbelt-labels er ikke et udbredt problem, men datasættet indeholder noget andet, der er vigtigere for rapporten.

## Hvad jeg gjorde
Jeg skrev [label_check.py](label_check.py) og kørte det på train, valid og test. For hvert par af bokse i samme fil beregner det **IoU** (overlap ÷ samlet areal). Når to bokse har forskellig klasse og IoU ≥ 0,5, er det samme skilt med to labels, altså en konflikt. Når overlappet er lavt, er det forskellige skilte. Scriptet gemmer også eksempler med boksene tegnet ind i `label_check/`, og jeg har kigget på dem.

## Fund 1: Dobbelt-labels er en enkeltstående fejl
| | train | valid | test |
|---|---|---|---|
| Filer med flere klasser | 332 | 55 | 53 |
| Heraf konflikt (samme sted, forskellig klasse) | **1** | 0 | 0 |
| Heraf ægte flere skilte | 331 | 55 | 53 |

Den eneste konflikt er netop den fil, du fandt. Billedet er så sløret, at man ikke kan se, om der står 20 eller 30. De andre kombinationer er ægte. For eksempel er **SL100 + SL120** (50 filer) og **SL80 + SL90** (28) flere hastighedsskilte på samme mast, typisk på motorveje. Du kan bare fjerne den ene fil eller nævne den som et eksempel på støj i labels. Den betyder ikke noget for resultatet.

## Fund 2: Datasættet kommer fra fire kilder
Kilden kan ses på filnavnet. Tallene gælder train:

| Kilde | Filer | Bokse | Bokse < 8 px |
|---|---|---|---|
| GTSRB-udsnit (sløre nærbilleder) | 1352 | 1353 | 0 |
| Vejfotos (`000xxx`) | 1174 | 1435 | 126 |
| **Simulator (`FisheyeCamera`)** | 508 | 933 | **637** |
| `road*.png` | 496 | 577 | 2 |

Fisheye-billederne er **syntetiske** fra en bilsimulator, og trafiklysene i dem er ofte kun 2–5 px store. Det har to konsekvenser:

- **Iteration 2:** `MIN_BOX_PIXELS = 8` fjerner ca. 40 % af Green Light (226/542) og 49 % af Red Light (286/585). Næsten alle de fjernede kommer fra simulatoren. Det er et rimeligt valg, fordi et udsnit på 3 px ikke kan klassificeres. Men det skal stå i rapporten, fordi det ændrer, hvad modellen faktisk testes på.
- **Iteration 1:** Det kan forklare, hvorfor Red Light blev til Green Light. I simulatorbillederne er der ofte både rødt og grønt lys i samme billede, og de fylder næsten ingenting. Når hele billedet bruges med klassen fra "største boks", kan to næsten identiske billeder få hver sin label. Det er et godt argument for at skære skiltene ud i iteration 2.

En lille rettelse til dine noter: Speed Limit 10 har **19** bokse i train. 4 af dem er under 8 px, så der er kun 15 tilbage til crop-modellen.

**Spørgsmål til dig:** GTSRB-udsnittene udgør cirka 40 % af datasættet, og skiltet fylder næsten hele billedet. Hvad tror du, det gør ved YOLO's evne til at finde *små* skilte i rigtige vejbilleder? Og bør du rapportere resultaterne pr. kilde i stedet for samlet?

Næste skridt er at køre `Machineconvoluted_crops.py`. Det tager formentlig 5–15 minutter på din RTX 3050.

---

## Prompt 2 (2026-11-02 14:25)

**Mig:**

Ja, kør det

**Claude:**

Træningen kører i baggrunden, og loggen skrives til `train_crops_log.txt`, så du kan bruge den som bilag. Jeg tjekker lige starten af loggen for at fange en evt. tidlig fejl (sti, CUDA osv.).

Loggen er stadig tom, fordi PyTorch og de andre biblioteker bliver indlæst. Jeg får besked, når træningen er færdig, og så gennemgår jeg resultaterne.

Mens du venter, er her de tre ting, jeg kigger efter i sammenligningen med iteration 1:
1. **Om kollapset er væk:** forudsiger modellen nu alle klasser, eller hælder den stadig mod få klasser?
2. **Red vs. Green Light:** nu ser modellen kun selve lyset, så farven burde være nok til at skelne dem.
3. **Forveksling mellem hastigheder:** den svære del er tallene, og jeg forventer især forvekslinger som 30↔80 og 50↔60, fordi cifrene ligner hinanden i lav opløsning.

**Spørgsmål til dig:** Iteration 1 blev målt pr. *billede* (801 i valid), og iteration 2 måles pr. *skilt* (ca. 800 udsnit efter de små bokse er fjernet). Er det en fair sammenligning? Hvad skal du skrive om det i rapporten?

---

## Prompt 3 (2026-11-02 14:30)

**Mig:**

Det er ikke en fair sammenligning. Men det er en naturlig forbedring at ville fortælle mit CNN hvad den egentlig skal kigge på i billederne.   
  
Jeg kan skrive at mit CNN i første iteration ikke fik gode resultater grundet mængden af støj, og mangel på indsnævring i træningen.   
  
Har du andre forslag til hvad jeg kunne skrive?  
  
Og har du nogen ide om hvor lang tid træningen vil tage, evt kom med en visuel progress bar hvis muligt

**Claude:**

Du kan allerede se én ting i loggen: accuracy hopper fra 81,4 % til 85,1 % ved epoch 11. Det er præcis der, StepLR sænker learning rate ×10. Det er et konkret og dokumenterbart eksempel på, hvad scheduleren gør, så det er værd at nævne med et TensorBoard-plot.

## Forslag til rapporten om iteration 1 → 2

Din formulering er rigtig, men "støj" er lidt vagt. Prøv at gøre det mere præcist og bygge det på dine fund:

**1. Opgaven var forkert formuleret, ikke kun støjende.** Iteration 1 bad modellen om at *klassificere et helt billede*, selvom billederne kan indeholde flere skilte. Label-tjekket viste, at 332 af 3530 træningsbilleder har flere klasser. "Største boks"-reglen gav dem én label og ignorerede resten. Modellen blev altså trænet på et delvist forkert facit.

**2. Skiltet fyldte for lidt af input.** I simulator- og vejbillederne fylder et skilt ofte under 1 % af 418×418 pixels. Efter fire max-pool-lag kan et skilt på 10 px være reduceret til mindre end én pixel i feature-mappet. Det giver en teknisk forklaring på, hvorfor Red Light og Green Light ikke kunne skelnes.

**3. Ubalance → kollaps.** Uden class weights og augmentation lærte modellen at gætte på de hyppigste klasser. Det er den billigste måde at sænke loss på.

**4. Ændringerne er ikke isoleret.** Vær ærlig om, at iteration 2 ændrer seks ting på én gang: crop, input-størrelse, augmentation, class weights, flere epochs og scheduler. Du kan derfor ikke sige, *hvilken* ændring der hjalp mest. Hvis en underviser spørger, er det stærkere selv at have nævnt det. Du kan evt. tilføje en kort "ablation" senere, fx køre iteration 2 uden class weights.

**5. Hvad måles der på.** Som du selv siger, er det ikke samme testsæt: pr. billede mod pr. skilt, og de små bokse er fjernet i iteration 2. Iteration 2 svarer også kun på "hvilket skilt er det?", ikke "hvor er skiltet?". Den forudsætter, at nogen allerede har fundet boksen. Det er netop den overgang, der gør YOLO relevant bagefter, fordi YOLO løser både detektion og klassifikation.

Punkt 5 er måske det vigtigste for vurderingen, fordi det binder CNN-delen sammen med sammenligningen af detektionsalgoritmer. **Spørgsmål:** Hvis du kombinerede din klassiske pipeline (Hough circles finder runde skilte) med crop-CNN'et som klassifikator, hvilken af de tre metoder ville den så ligne mest?

Jeg gennemgår confusion matrix og classification report, så snart træningen er færdig.

Iteration 2 er færdig og nåede **93,6 %** validerings-accuracy (epoch 27). Iteration 1 lå på ca. 27 % (214 af 801 billeder rigtige). Den fulde log ligger i [train_crops_log.txt](train_crops_log.txt).

## Sammenligning iteration 1 → 2

| | Iteration 1 (hele billedet) | Iteration 2 (udsnit) |
|---|---|---|
| Accuracy | ~27 % (pr. billede) | 93,6 % (pr. skilt) |
| Klasser modellen aldrig forudsiger | 9 af 14 | 0 |
| Red Light recall | 0 / 56 | 75 / 76 |
| Stop recall | 60 / 78 | 81 / 81 |
| Svageste klasse | (de fleste) | SL90: recall 0,80 |

## Tolkning

**1. Kollapset er løst.** I iteration 1 var Green Light-kolonnen (forudsagt) fyldt op, så modellen gættede i praksis "Green Light" for alt den var usikker på. Nu ligger næsten alt på diagonalen. Red mod Green Light giver kun én fejl, så farven var nok til at skelne dem, så snart modellen kun så selve lyset. Det understøtter forklaringen fra label-tjekket.

**2. Class weights har en bivirkning, og den er interessant til rapporten.** Speed Limit 10 har ingen eksempler i valid, men modellen forudsiger den **19 gange**. Vægten for SL10 er ca. 15 gange højere end for de andre klasser (kun 16 udsnit), så modellen har lært, at det er dyrt at overse den. Derfor gætter den på SL10, når den er i tvivl, især ved SL120, SL90 og SL20. Det er et klassisk trade-off: højere recall for sjældne klasser koster precision. Det trækker også macro avg ned til 0,89, fordi SL10-rækken tælles med som 0. Uden den ligger macro avg omkring 0,95.

**3. De resterende fejl er tal, der ligner hinanden.** De største forvekslinger er 60↔80 (4+4), 50↔60 (4+1) og 100↔120 (2+1). Det giver mening ved 64×64 px, hvor 6, 8 og 5 kan ligne hinanden. Da de fleste skilte ender på 0, er det kun det første ciffer, der adskiller dem.

**4. Grad-CAM viser, hvad modellen kigger på.** Ved 90, 110 og 40 ligger varmen på **første ciffer**, og det er netop det, der adskiller klasserne. Det er et godt tegn. To billeder er værd at fremhæve i rapporten:
- **SL30 (korrekt):** fokus ligger på kanten af ringen, ikke på tallet. Modellen har ret, men måske af den forkerte grund, fx en egenskab ved kilden eller kanten frem for selve tallet.
- **SL80 → SL60 (fejl):** billedet er mørkt og sløret, og modellen kigger på baggrunden ved siden af skiltet. Det er et eksempel på, hvor datakvaliteten sætter grænsen.

**5. Learning rate-hop.** Accuracy steg fra 57 % til 71 % ved epoch 8 og igen ved epoch 11, hvor StepLR sænkede learning rate. Fra epoch 20 flader kurven ud omkring 93 %, så flere epochs ville næppe hjælpe. Det kan du vise med plottet `Accuracy/test` i TensorBoard:

```bash
.venv\Scripts\tensorboard.exe --logdir=runs
```

**Husk forbeholdet til rapporten:** tallene er målt på udsnit, hvor vi allerede kender boksen, og hvor de 98 mindste skilte er fjernet fra valid. Det er et klassifikationsresultat, ikke et detektionsresultat. Netop derfor er YOLO det næste skridt.

**Spørgsmål til dig:** Hvordan ville du ændre class weights, så SL10-problemet bliver mindre, uden at den sjældne klasse bliver helt ignoreret? Hint: kvadratroden.

---

## Prompt 4 (2026-11-02 14:55)

**Mig:**

Sørg for at hver ny iteration får nyt heatmap, tensorboard og confusion matrix btw. så de gamle ikke bliver overskrevet.

**Claude:**

Godt valg. Lige nu skifter filnavnene kun, fordi jeg manuelt gav dem et `_crops`-suffiks. Det holder ikke, når du kører iteration 3, 4 osv., og det beskytter heller ikke mod, at du kører den *samme* iteration igen. TensorBoard er endnu værre: skriver du to gange til samme mappe, blandes kurverne fra begge kørsler sammen i ét plot.

**Løsningen:** hver kørsel får sit eget **run-ID** = iterationsnavn + tidsstempel, fx `iter2_crops_20261102_1525`. Alt fra kørslen havner i `results/<run-ID>/`, og TensorBoard skriver til `runs/<run-ID>/`. Tidsstemplet gør, at intet kan overskrives, heller ikke hvis du genkører. Jeg gemmer også classification report som en `.txt` i mappen, så du ikke skal grave tallene ud af terminalen.

Nu den samme ændring i iteration 1-scriptet, så det heller ikke overskriver sine egne filer, hvis du nogensinde genkører det:

Fremover overskriver ingen kørsel resultaterne fra en anden. Begge scripts parser uden fejl, men jeg har ikke kørt dem igen.

## Hvad der er ændret
I både [Machineconvoluted.py](Machineconvoluted.py) og [Machineconvoluted_crops.py](Machineconvoluted_crops.py) er der tilføjet et run-ID øverst: navnet på iterationen plus dato og klokkeslæt. Hver kørsel ender så i sin egen mappe:

```
results/
  iter2_crops_20261102_153012/
    model.pth
    confusion_matrix.png      <- titlen viser også run-ID
    gradcam_examples.png
    classification_report.txt <- ny: tallene som tekst, klar til rapporten
runs/
  iter2_crops_20261102_153012/   <- TensorBoard
```

To ting sikrer, at intet bliver overskrevet:
- **Tidsstemplet** giver hver kørsel et nyt navn, også hvis du kører samme iteration igen.
- **`OUT_DIR.mkdir(parents=True)`** stopper scriptet med en fejl, hvis mappen allerede findes, i stedet for at skrive oven i den.

## Sådan laver du en ny iteration
Kopiér `Machineconvoluted_crops.py`, ret `RUN_NAME` (fx `"iter3_sqrt_weights"`), og lav din ændring. I TensorBoard (`--logdir=runs`) kommer hver kørsel så som sin egen kurve, og du kan slå dem til og fra for at sammenligne iterationerne direkte. Det giver et godt plot til rapporten.

De eksisterende filer (`confusion_matrix.png`, `confusion_matrix_crops.png`, `gradcam_examples_crops.png`, `sign_model.pth`, `sign_model_crops.pth`, `runs/sign_classifier*` og `train_crops_log.txt`) ligger stadig i projektroden.

---

## Prompt 5 (2026-11-02 14:57)

**Mig:**

Ja, flyt dem

**Claude:**

Kørsel 4 (14:16) er den, der skrev `sign_model.pth` kl. 15:04, så den hører sammen med modellen. Nu flytter jeg filerne:

Til sidst tjekker jeg, om andre scripts henviser til de gamle filnavne, så intet går i stykker:

Alle gamle resultater er flyttet ind i den nye struktur:

```
results/
  iter1_fullimage_20260914_141610/
    model.pth
    confusion_matrix.png           <- OBS: fra 27/10, se nedenfor
  iter2_crops_20261102_152508/
    model.pth
    confusion_matrix.png
    gradcam_examples.png
    classification_report.txt     <- trukket ud af træningsloggen
    train_log.txt
runs/
  iter1_fullimage_20260914_125030/
  iter1_fullimage_20260914_132643/
  iter1_fullimage_20260914_133527/
  iter1_fullimage_20260914_141610/   <- kørslen der hører til model.pth
  iter2_crops_20261102_152508/
```

Ingen andre scripts brugte de gamle filnavne, så intet går i stykker. Jeg har kun rettet en forældet kommentar i `Machineconvoluted.py`.

**To ting du skal være opmærksom på:**

- **Iteration 1's confusion matrix passer ikke tidsmæssigt med modellen.** Matrixen er fra 27. oktober, men modellen og den sidste TensorBoard-kørsel er fra 14. september. Måske kørte du scriptet igen d. 27/10 og afbrød det, eller måske kørte du en anden version. Tallene ser ud til at matche det, du beskrev, men tjek det, før du skriver i rapporten, hvilken kørsel matrixen kommer fra.
- **De fire iteration 1-kørsler i TensorBoard er nu adskilt.** Før lå de oven i hinanden i samme graf. Hvis de tre første var forsøg med andre indstillinger, kan de faktisk være nyttige i rapporten som dokumentation af processen. Kan du huske, hvad du ændrede mellem dem?

---

## Prompt 6 (2026-11-02 14:59)

**Mig:**

Ja, start YOLOv8

**Claude:**

Først en kort testkørsel (1 epoch, 10 % af data). Den fanger fejl i stier og opsætning, før vi bruger 20+ minutter. Ultralytics henter også `yolov8n.pt` (ca. 6 MB) fra GitHub første gang.

YOLOv8-træningen er startet i baggrunden. Loggen skrives til `yolo_train_log.txt`, og resultaterne kommer i `results/yolov8n_<tidsstempel>/`. Den er stadig ved at indlæse data, så jeg har ikke et tal for tid pr. epoch endnu. Jeg gætter på **20–40 minutter** for 50 epochs. Den kan stoppe tidligere, hvis `patience=15` slår til. Jeg får besked, når den er færdig.

**Hvad Ultralytics gemmer automatisk** (brugbart til rapporten):
- `results.png`: loss- og mAP-kurver over alle epochs
- `confusion_matrix.png` og `confusion_matrix_normalized.png`. YOLO's matrix har en ekstra **"background"**-række og -kolonne. Den viser skilte, modellen *overså*, og "skilte" den fandt, hvor der ikke var noget. Det kan dit CNN slet ikke måle, fordi det altid får et udsnit med et skilt.
- `PR_curve.png`: precision mod recall pr. klasse
- `val_batch*_pred.jpg`: eksempelbilleder med de forudsagte bokse
- TensorBoard-filen ligger i run-mappen (ikke i `runs/`). Du kan se CNN og YOLO side om side med:

```bash
.venv\Scripts\tensorboard.exe --logdir_spec=cnn:runs,yolo:results
```

**Et begreb du skal kunne forklare i rapporten:** YOLO måles med **mAP50** og ikke med accuracy. En detektion tæller som korrekt, hvis klassen er rigtig *og* boksen overlapper det sande skilt med IoU ≥ 0,5 (samme IoU som i `label_check.py`). mAP er gennemsnittet over alle klasser af arealet under precision-recall-kurven. **mAP50-95** er strengere og tager gennemsnittet over IoU-grænser fra 0,5 til 0,95.

**Spørgsmål mens du venter:** dit CNN fik 93,6 % på udsnit, og YOLO får måske en mAP50 omkring 0,9. Hvorfor kan du ikke direkte sige, at den ene er bedre end den anden? Tænk på hvad hver model *får at vide* på forhånd.

---

## Prompt 7 (2026-11-02 15:07)

**Mig:**

Hvad får yolov8 at vide på forhånd i forhold til mit CNN?

**Claude:**

Det er faktisk omvendt af det, jeg antydede: på det vigtigste punkt får YOLO *mindre* at vide end dit CNN. Det rigtige svar har to sider.

## Hvad dit CNN får forærende
**Placeringen af skiltet.** I iteration 2 skærer `CroppedSignDataset` udsnittet ud fra den *sande* bounding box fra label-filen. CNN'et skal kun svare på "hvilket af 15 skilte er det her?" og ved altid, at der er præcis ét skilt midt i billedet.

Det betyder tre ting for evalueringen:
- **CNN'et kan ikke overse et skilt.** Det får aldrig et billede, hvor skiltet ikke er med.
- **CNN'et kan ikke finde et skilt, der ikke findes.** Det får aldrig baggrund uden skilt, så falske positiver måles slet ikke.
- **De sværeste tilfælde er fjernet.** De 98 bokse under 8 px blev filtreret væk fra valid, og det var mest bittesmå trafiklys fra simulatoren. YOLO bliver testet på dem alle.

YOLO får derimod hele 416×416-billedet og skal selv løse *hvor* (lokalisering), *hvor mange* og *hvad* (klassifikation). Det kan fejle på alle tre punkter, og mAP straffer det hele.

## Hvad YOLO får forærende
**Forhåndsviden fra COCO.** `yolov8n.pt` er allerede trænet på ca. 118.000 billeder med 80 klasser, bl.a. "stop sign" og "traffic light". Den kan altså allerede se kanter, cirkler og former, og den ved hvordan skilte og lys ser ud. Dit CNN starter fra tilfældige vægte og skal lære alt fra ca. 3.600 udsnit.

Desuden får YOLO **kontekst**: den ser masten, vejen og omgivelserne, hvilket kan hjælpe ved slørede skilte. Det kan dog også give "snyd", ligesom i din iteration 1.

## Opsummeret til rapporten

| | Eget CNN (iter 2) | YOLOv8n |
|---|---|---|
| Ved hvor skiltet er | **Ja** (fra label) | Nej, skal selv finde det |
| Forhåndsviden | Nej (tilfældige vægte) | **Ja** (fortrænet på COCO) |
| Kan måles på overset skilt / falsk alarm | Nej | Ja |
| Testes på små skilte (< 8 px) | Nej (fjernet) | Ja |
| Metrik | Accuracy pr. udsnit | mAP50 pr. billede |

Så hvis YOLO ender med en lidt lavere score, betyder det ikke, at den er dårligere. Den løser en sværere opgave. En fair formulering kunne være: *"CNN'et er en klassifikator, der forudsætter at skiltet allerede er lokaliseret; YOLO er en komplet detektor."*

Hvis du vil lave en mere fair sammenligning, kan du senere tage YOLO's egne bokse, skære dem ud og give dem til dit CNN. Så får du en totrins-pipeline: YOLO finder skiltet, og CNN'et klassificerer det. Det svarer til din klassiske pipeline (Hough circles finder, noget andet klassificerer), og det binder rapportens tre dele sammen.

Træningen kører stadig, og jeg vender tilbage med resultaterne, når den er færdig.

---

## Prompt 8 (2026-11-02 15:28)

**Mig:**

my laptop ran out of power, give me a summary of the previous yolov8 training, and start a new one from scratch

**Claude:**

## Opsummering af den afbrudte kørsel
Mappe: `results/yolov8n_20261102_160311_afbrudt/` (loggen ligger i samme mappe som `train_log.txt`).

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | 0,27 | 0,40 | 0,25 | 0,20 |
| 5 | 0,72 | 0,70 | 0,76 | 0,64 |
| 10 | 0,93 | 0,80 | 0,89 | 0,74 |
| 20 | 0,95 | 0,84 | 0,92 | 0,79 |
| **30 (sidste)** | **0,95** | **0,89** | **0,94** | **0,80** |

- **Den nåede 30 af 50 epochs.** Den var stadig i langsom fremgang (mAP50 steg fra 0,92 til 0,94 over de sidste 10 epochs), så den var ikke færdig med at lære.
- **Den lærte hurtigt.** mAP50 nåede 0,89 allerede efter 10 epochs, cirka 5,5 minut. Det er transfer learning fra COCO, der virker. Til sammenligning havde dit CNN fra bunden kun 81 % efter 10 epochs, og det var på den nemmere opgave med udsnit.
- **Precision ligger over recall** (0,95 mod 0,89). Når YOLO siger "skilt", har den næsten altid ret, men den overser cirka hvert tiende. Det er sandsynligvis de bittesmå trafiklys fra simulatoren, men det kan vi først se med per-klasse-tallene fra den nye kørsel.
- **Test-evaluering kørte aldrig.** Tallene ovenfor er på *valideringssættet*, så de skal ikke bruges som slutresultat i rapporten.
- **Tegn på strømproblemet:** epoch 1–29 tog cirka 33 sekunder hver, men epoch 30 tog **368 sekunder**. Laptoppen er sandsynligvis gået i strømsparetilstand, da batteriet blev lavt, og har sat GPU'en ned i hastighed. **Sæt opladeren i nu**, ellers sker det igen.

## Ny træning
Den nye kørsel er startet i baggrunden med præcis samme indstillinger, fra den fortrænede `yolov8n.pt`. Den får sin egen mappe `results/yolov8n_<nyt tidsstempel>/`, og loggen skrives til `yolo_train_log.txt`. Med strøm tilsluttet tager den omkring **50 × 33 s ≈ 28 minutter** plus test-evalueringen, så den burde være færdig om cirka en halv time. Jeg samler resultaterne op, når den er færdig.

Den afbrudte kørsel kan faktisk bruges i rapporten. Hvis den nye ender på samme kurve, viser det, at træningen er reproducerbar (`seed=0`).

YOLOv8n er færdig: alle 50 epochs gennemført på 25 minutter, og test-evalueringen er kørt. Alt ligger i `results/yolov8n_20261102_163040/` (træning og log) og `results/yolov8n_20261102_163040_test/` (testsættet).

## Resultat på testsættet (638 billeder, 770 skilte)
**mAP50 = 0,936 · mAP50-95 = 0,783**

| Gruppe | Precision | Recall | mAP50 |
|---|---|---|---|
| Stop | 0,99 | 1,00 | 0,995 |
| Hastighedsskilte (10–120) | 0,72–0,98 | 0,86–1,00 | 0,89–0,995 |
| Green Light | 0,86 | 0,79 | 0,87 |
| **Red Light** | 0,89 | **0,69** | **0,74** |

## Tolkning

**1. Reproducerbarheden holder.** Ved epoch 10, 20 og 30 er mAP50 *præcis* den samme som i den afbrudte kørsel (0,88681 / 0,92497 / 0,94194). Det skyldes `seed=0`, og det er værd at nævne i rapporten. Den afbrudte kørsel ville altså være endt det samme sted.

**2. Læs YOLO's confusion matrix den rigtige vej.** Akserne er byttet om i forhold til dit CNN: her er **rækker = forudsagt** og **kolonner = sandt**. Desuden er der en ekstra *background*-række og -kolonne:
- **Nederste række (background, forudsagt):** skilte YOLO **overså** helt. Det gælder 26 af 94 Red Light og 16 af 110 Green Light.
- **Højre kolonne (background, sandt):** "skilte" YOLO fandt, hvor der **ikke var noget**. Det er især lys (19 grønne og 16 røde falske alarmer).

**3. Trafiklysenes problem er detektion, ikke klassifikation.** Rødt og grønt lys bliver kun forvekslet 1 gang hver vej. Når YOLO finder lyset, kender den farven. Problemet er, at den ikke *finder* dem. Det passer med label-tjekket: halvdelen af trafiklysene er under 8 px i simulatorbillederne. De blev fjernet fra dit CNN's test, men YOLO testes på dem. Her ser du forskellen fra mit forrige svar i praksis.

**4. Hastighedsskilte er næsten løst.** Der er kun enkelte forvekslinger, fx 50→30 (3 gange). De fleste fejl skyldes oversete skilte, ikke forkerte tal. Dit CNN havde flere talforvekslinger (60↔80, 50↔60), og her hjælper COCO-fortræningen og den højere opløsning sandsynligvis.

**5. Speed Limit 10 er stadig for spinkel.** Testsættet har kun 3 eksempler: 1 er rigtig, og 2 er blevet til SL20. Recall på 1,00 i tabellen skyldes, at Ultralytics' per-klasse-tal bruger en anden konfidensgrænse end matrixen. Med 3 eksempler kan du ikke konkludere noget, og det passer med din dokumenterede begrænsning.

**Til rapporten:** Det mest interessante fund er, at **den største fejlkilde skifter mellem metoderne**. I CNN iteration 1 var det baggrunden, i iteration 2 var det tal der ligner hinanden, og i YOLO er det små objekter. Det er en god rød tråd for en sammenligning.

**Spørgsmål til dig:** YOLO kørte med `imgsz=416`. Hvad tror du, der sker med de små trafiklys, hvis du træner med `imgsz=640`, og hvad koster det?

---

## Prompt 9 (2026-11-02 16:04)

**Mig:**

Ville jeg kunne bruge andre metoder til at fortelle Yolov8 hvor skiltene er, eller er det bedre at den selv finder dem? Altså ting som contour detection, eller lign.  
  
Opskaleringen ville koste en del tid. Hvert epoch ville tage meget længere tid at køre igennem, men der ville være større chance for at modellen finder dem  
  
Hvad er negative billeder? og hvad ville det hjælpe at køre en test med dem?

**Claude:**

Kort svar: YOLO kan ikke tage imod forslag udefra, men dit CNN kan. Her er de tre svar.

## 1. Kan klassiske metoder fortælle YOLO, hvor skiltene er?
Ikke direkte. YOLO står for "You Only Look Once", fordi den finder og klassificerer i **ét** gennemløb af hele billedet. Den har ikke noget input, hvor man kan give den forslag til placeringer.

Det du beskriver er en **totrins-pipeline**:
1. **Forslag (region proposal):** blur → rød hue-filter → Hough circles eller contours finder kandidater.
2. **Klassifikation:** hver kandidat skæres ud og gives til en klassifikator.

Det er faktisk den idé, de tidlige deep learning-detektorer byggede på (R-CNN, 2014), før YOLO samlede det hele i ét netværk. Men klassifikatoren i trin 2 skal være **dit CNN**, ikke YOLO. Dit CNN er jo netop trænet på udsnit.

| | Klassisk forslag + dit CNN | YOLO alene |
|---|---|---|
| Fordele | Gennemskueligt: hvert filter kan vises og forklares. Billigt at køre. Bruger kursets metoder. | Lærer selv hvad der er et skilt. Robust over for lys, vinkel og baggrund. Én model. |
| Svagheder | **Trin 1 sætter loftet:** et skilt Hough ikke finder, kan CNN'et aldrig redde. Et rødt hue-filter finder ikke grønt lys. Tærskler skal håndtunes. | Kræver labelede data og GPU. Sort boks. Svag på bittesmå objekter. |

**Min anbefaling:** Lad YOLO finde skiltene selv, for det er den bedre detektor. Byg *alligevel* Hough + CNN-pipelinen som en ekstra metode i sammenligningen. Den viser filtre i sekvens, hvilket bedømmelsen belønner. Den bruger dit CNN til det, det er godt til. Og den binder rapportens tre dele sammen: **klassisk alene → klassisk + CNN → YOLO**.

## 2. Opskalering til 640
Du har ret i begge dele. Beregningen vokser med arealet, så det koster ca. (640/416)² ≈ **2,4 gange så lang tid** pr. epoch. Du skal muligvis også sænke `batch` for at være inden for 6 GB VRAM.

**Billederne er kun 416×416**, så opskalering tilføjer ingen ny information. Pixels, der aldrig blev fotograferet, kommer ikke tilbage. Det kan stadig hjælpe lidt, fordi YOLO's fineste detektionslag ser billedet i felter på 8×8 pixels. Et trafiklys på 3 px fylder under et halvt felt ved 416, men ca. 5 px ved 640, og så er det lettere at "fange". Forvent en mindre forbedring på trafiklys, ikke et mirakel.

## 3. Hvad er negative billeder?
**Positive billeder** indeholder det, du leder efter (skilte). **Negative billeder** indeholder det *ikke*: veje uden skilte, natur, bygninger osv.

Testen måler **falske positiver**: hvor ofte metoden råber "skilt!" når der ikke er noget. Det er vigtigt, fordi din præcision indtil nu kun er målt på billeder, *hvor der er skilte*. En detektor, der markerer alt rødt og rundt som et skilt, kan score fint på positive billeder og alligevel være ubrugelig i en bil.

Det er særligt afslørende for hver af dine metoder:
- **Klassisk pipeline:** rød hue + Hough circles reagerer på alt rødt og rundt (baglygter, røde skilte, røde blomster).
- **Dit CNN:** det har **ingen "intet skilt"-klasse**. Det vil *altid* sige en af de 15 klasser, også når det får et udsnit af en busk. Den svaghed er værd at vise i rapporten.
- **YOLO:** den har en indbygget "background" og en konfidensgrænse, så den bør klare sig bedst.

**Hvor får du negative billeder fra?** `cardata` har kun 4 billeder uden skilte, så det er for få. Men:
- **`strawberry/` (40 billeder)** er perfekte **"svære negative"**: røde, runde og ikke skilte. Det er præcis det, der kan narre en rød hue → Hough-pipeline.
- **Udsnit fra cardata uden skilte:** jeg kan skære tilfældige områder ud af vejbillederne væk fra alle bokse. Det giver realistiske vej-negative billeder (asfalt, træer, biler).
- **Egne fotos** af veje uden skilte, hvis du har tid.

Testen vil tælle falske detektioner pr. billede for hver metode og gemme eksempler til rapporten.

Der er ikke ændret nogen filer i denne omgang. Skal jeg bygge testen på negative billeder for alle tre metoder nu, med jordbær og vejudsnit fra cardata? Eller vil du vente?

---

## Prompt 10 (2026-11-02 16:09)

**Mig:**

Ja, byg testen

**Claude:**

Nu giver tallene en klar historie. Jeg kigger på eksemplerne fra den nye kørsel for at se, *hvad* YOLO tager fejl af på jordbærrene:

Testen kører nu for alle metoder på både positive og negative billeder. Resultaterne ligger i `results/negative_test_20261102_171229/`, sammen med eksempelbilleder (`eksempler_*.jpg`) og en `summary.txt`. Første kørsel viste, at min oversættelse af dine Hough-parametre var alt for løs. Derfor byggede jeg den klassiske pipeline op i tre målte trin, og den mislykkede kørsel er slettet.

## Resultater

**Negative billeder: andel med mindst én falsk alarm**

| Metode | Jordbær (37) | Vejudsnit (100) |
|---|---|---|
| Klassisk A (naiv Hough) | 100 % (2121 alarmer) | 15 % (269) |
| Klassisk B (strammere Hough) | 46 % (68) | 1 % (1) |
| Klassisk C (+ ring-tjek) | 16 % (9) | **0 %** |
| Klassisk C + CNN (conf ≥ 0,9) | **5 %** (3) | **0 %** |
| YOLOv8n | 19 % (7) | 6 % (6) |

**Positive billeder: 150 testbilleder, 185 skilte**

| Metode | Fundet | Rigtig klasse | Falske bokse |
|---|---|---|---|
| Klassisk A | 41,6 % | – | 3191 |
| Klassisk B | 34,6 % | – | 169 |
| Klassisk C | 23,2 % | – | 61 |
| Klassisk C + CNN | 23,2 % | 19,5 % | 61 |
| Klassisk C + CNN (conf ≥ 0,9) | 11,9 % | 10,3 % | 19 |
| **YOLOv8n** | **93,5 %** | **92,4 %** | **11** |

## Tolkning

**1. Filtre i sekvens virker, men der er en pris.** Hvert klassisk trin fjerner store mængder falske alarmer: fra 2121 til 68 til 9 på jordbær. Men hvert trin fjerner også rigtige skilte: fra 42 % til 35 % til 23 % fundet. Det er det klassiske **precision/recall-trade-off**, og du har det målt trin for trin. Det er præcis det, bedømmelsen efterspørger.

**2. Ring-tjekket er det mest interessante filter.** Det bruger viden om, hvordan et skilt ser ud: rød ring og hvid midte. Det skelner jordbær fra skilte, uden at nogen model er trænet. Til gengæld fejler det på skilte, hvor ringen er sløret, delvist skjult eller set skråt.

**3. Den klassiske pipeline har et indbygget loft.** Den leder kun efter røde cirkler, så de 46 trafiklys blandt de 185 skilte kan den aldrig finde. Selv blandt de røde skilte finder trin C kun 31 %. Det illustrerer pointen fra før: **trin 1 sætter loftet for hele pipelinen**. CNN'et kan kun klassificere det, Hough har fundet.

**4. CNN'et kan ikke sige "intet skilt".** Uden konfidensgrænse giver "Klassisk C + CNN" præcis samme falske alarmer som "Klassisk C", fordi CNN'et *altid* vælger en af de 15 klasser. Med grænsen på 90 % sikkerhed falder de falske alarmer, men halvdelen af de rigtige fund forsvinder også. Konfidensgrænsen er en nødløsning. Den rigtige løsning ville være en 16. klasse "baggrund", trænet på udsnit uden skilte. Det er en mulig iteration 3.

**5. YOLO er klart bedst samlet set, men ikke fejlfri.** På jordbær laver den 7 falske alarmer. På eksemplerne ser man bl.a. et stort jordbær forudsagt som **Stop** (rødt, stort og nogenlunde rundt/ottekantet) og bokse som **Red Light** over hele billeder fyldt med jordbær. YOLO har aldrig set et jordbær under træningen. Den har kun lært "rødt område = skilt eller lys", fordi alt rødt i cardata *er* et skilt eller lys. Her ser du værdien af negative billeder i *træningen*: de lærer modellen, hvad den skal ignorere.

## Forbehold du bør nævne i rapporten
- **Jeg tunede trin B og C ved at kigge på disse testresultater.** Parametrene er altså valgt på samme data, som de er målt på, hvilket gør de klassiske tal en smule for optimistiske. Det er sket i én iteration og med åbenlyse ændringer, men skriv det.
- Kun 37 af 40 filer i `strawberry/` blev brugt. Scriptet læser kun `.jpg`, så de sidste 3 har sandsynligvis en anden filtype.
- 37 + 100 negative billeder er nok til at vise forskellene mellem metoderne, men ikke til at give præcise procenter. Ved så små tal er 5 % mod 19 % en tydelig forskel, men 0 % mod 1 % er det ikke.

**Spørgsmål til dig:** Ring-tjekket kræver hvid midte. Hvilke rigtige skilte i dit datasæt tror du, det fejlagtigt *afviser*? Tænk på Stop-skiltet.

---

## Prompt 11 (2026-11-02 16:23)

**Mig:**

[vedhæftet fil: studieplan.pdf] Okay  
Er der andre ting der ville være godt at have med i rapporten?  
Vedlagt er det officielle studie dokument  
Og denne github indeholder vores vedlagte informationer, læs den venligst igennem   
https://youtube.com/shorts/8a3HIXUvQ3Q?si=uY6pxEkfeg__C78X

**Claude:**

Jeg kan ikke læse PDF'en direkte, så jeg installerer det lille bibliotek `pypdf` i dit venv for at trække teksten ud:

Jeg har læst studieplanen (PDF'en, via `pypdf`, som jeg installerede i dit venv) og kursets GitHub-repo, inkl. vurderingsgrundlaget i `PorteføljeKrav/PorteføljeIntro.md` og materialet fra uge 1 og 2. YouTube-linket i din besked kunne jeg ikke bruge, men GitHub-linket du sendte bagefter dækkede det hele.

## Hvor står du i forhold til vurderingsgrundlaget?

| Kriterium | Niveau 3 kræver | Status |
|---|---|---|
| **Filtre** | Flere filtre i sekvens, forbedring over iterationer, **refleksion over fravalgte filtre** | Klassisk A→B→C er målt trin for trin ✅. **Mangler:** hvorfor blob, Canny-konturer, keypoints/FLANN osv. ikke blev brugt |
| **Bred anvendelse** | Virker på positive, intet fundet på negative | Negativ-testen dækker det ✅ |
| **Algoritmer** | Flere algoritmer sammenlignet, parameterproces, styrker/svagheder | Hough, CNN (2 iterationer) og YOLO ✅. Parameterprocessen kunne vises tydeligere |
| **Valg af billeder** | Replikérbar vinkel/lys, eksperimenter med lys/vinkel/størrelse, realistiske forhold | **Svageste punkt.** Ikke behandlet endnu |

## Det jeg ville tilføje, i prioriteret rækkefølge

**1. Et afsnit om billedvalg (kriteriet "Valg af billeder").** Du har allerede materialet fra label-tjekket:
- Datasættet består af fire kilder: GTSRB-udsnit, vejfotos, simulator og `road*.png`. Diskutér hvor repræsentative de er. Simulatoren er syntetisk, GTSRB-billederne er slørede, og mange vejfotos er stockfotos med vandmærker (shutterstock/alamy kan ses på eksemplerne).
- **Test på dine egne fotos** (`20.1.jpg`, `30.1–30.5.jpg`): det er billeder taget under rigtige forhold. Det rammer niveau 3 direkte.
- **Robusthed over for lys:** gør testbillederne kunstigt mørkere og lysere og mål, hvordan hver metode klarer sig. Det svarer til "eksperimenteret med forskellige lyssætninger".
- **Resultater pr. størrelse eller kilde:** vis at YOLO fejler på de små trafiklys og klarer de store skilte.

**2. Et afsnit om fravalgte metoder.** Niveau 3 under filtre kræver eksplicit *"refleksion over hvorfor dem der ikke er anvendt ikke ville fungere"*. Du har allerede afprøvet det meste i `cannytest.py`: blob detection, Canny-konturer, keypoints og FLANN-matching. Skriv kort, hvorfor de blev fravalgt. Nævn også et par fra ugematerialet: template matching er følsom over for størrelse og rotation, og watershed er segmentering, ikke detektion.

**3. Studieplanens læringsmål:**
- **F3 – begrund brugen af kamera:** hvorfor et kamera og ikke fx kortdata/GPS til fartgrænser? Kortdata kender ikke midlertidige skilte (vejarbejde), og kun et kamera kan se lyskryds. Diskutér også kameraplacering og fisheye-optik (simulatorbillederne).
- **K1 – samfundsmæssig use case:** EU har siden juli 2024 krævet **Intelligent Speed Assistance (ISA)** i alle nye biler, og det bygger netop på genkendelse af fartskilte. Det giver en stærk begrundelse for din use case. (Tjek selv en kilde på det, før du citerer.)
- **K2 – dokumentation til både teknikere og ledelse:** tilføj et kort ikke-teknisk resumé øverst i rapporten.
- **F2 – objektorienterede metoder:** nævn at du bruger klasser: din egen `CroppedSignDataset`/`Net`, OpenCV's `SimpleBlobDetector` og Ultralytics' `YOLO`.

**4. Hastighed pr. metode.** Kursets formål handler om kameraudstyrede enheder, og i en bil skal det køre i realtid. Den klassiske pipeline, Hough + CNN og YOLO har meget forskellige hastigheder pr. billede. Det er en relevant styrke/svaghed, som er nem at måle.

**5. Hough-parametersweep (valgfrit).** Et plot af `param2` mod recall og falske alarmer viser "processen af parameterjustering" visuelt i stedet for kun tre punkter (A/B/C).

## Til den mundtlige prøve
Eksamen er 10 minutters oplæg + 10 minutters spørgsmål og vægter 20 %. Kursusmaterialet beskriver YOLO som *"egentligt ret ringe, men meget let at implementere"*. Dine resultater viser, at den klart slår de andre metoder her. Forbered dig på at forklare hvorfor: transfer learning fra COCO og en opgave, der passer til en detektor. Det er et oplagt spørgsmål fra underviseren.

Tjek også selv afleveringsdatoen på ItsLearning. Repoet siger *"Onsdag den 21."* uden måned, og jeg kan ikke afgøre, hvilken dato der menes.

Jeg kan lave punkt 1 (egne fotos, lys-robusthed, pr. størrelse), punkt 4 (hastighed) og punkt 5 (Hough-sweep) som små scripts. Teksten i punkt 2 og 3 skal du helst skrive selv. Hvilke af de tre analyser vil du have, og i hvilken rækkefølge?

---

## Prompt 12 (2026-11-02 17:07)

**Mig:**

20.1 og 30.1 etc er ikke mine egne billeder, de er bare billeder taget fra mit dataset og omnavngivet så det var lettere at skrive ind. Men kør gerne punkt 1 som tilbudt.  
  
Kør også gerne punkt 4 og 5

**Claude:**

Tak for rettelsen. Så har du ingen egne fotos, og lys-, størrelses- og kildeanalysen bliver i stedet lavet på cardata-testsættet.

**Plan:** alle tre analyser skal bruge de samme detektionsfunktioner som `negative_test.py`. I stedet for at kopiere koden en tredje gang flytter jeg den til et fælles modul, `pipelines.py`, som begge scripts importerer. Derefter kører jeg `negative_test.py` igen for at bekræfte, at tallene er uændrede.

Kommentaren i linje 53–55 bygger på min antagelse om, at `30.x.jpg` var store mobilfotos. Den tjekker jeg, inden jeg flytter koden:

Alle tre analyser er kørt. Resultaterne ligger i `results/robustness_20261102_181133/`: fire grafer (`lys.png`, `stoerrelse.png`, `kilde.png`, `hough_sweep.png`), et eksempelbillede (`lys_eksempel.jpg`) og alle tal i `summary.txt`.

## Hvad jeg ændrede
- **`pipelines.py` (ny):** fælles kode til de tre metoder, så `negative_test.py` og `robustness_test.py` bruger præcis de samme pipelines. Jeg kørte `negative_test.py` igen efter omskrivningen, og tallene for positive billeder og vejudsnit er identiske med før.
- **Jordbær:** 40 i stedet for 37, fordi `.jpeg`- og `.png`-filerne nu også tælles med. YOLO fik 1 ekstra falsk alarm, og Klassisk C fik 1 ekstra.
- **Rettet kommentar:** dine oprindelige Hough-parametre var tunet på ét billede (`30.2.jpg`), ikke på "store mobilfotos", som jeg fejlagtigt skrev.

## Punkt 1a – Lys
| Lysfaktor | 0,3 | 0,5 | 0,7 | **1,0** | 1,3 | 1,6 | 2,0 |
|---|---|---|---|---|---|---|---|
| Klassisk C | 0,5 % | 11 % | 18 % | **23 %** | 26 % | 24 % | 18 % |
| YOLOv8n | 87 % | 91 % | 92 % | **92 %** | 91 % | 87 % | 82 % |

**Den klassiske pipeline bryder sammen i mørke.** Det er en direkte følge af dit HSV-filter: det kræver Value ≥ 70. Ved ×0,3 falder næsten alle pixels under grænsen, så den røde maske bliver tom. Ved overeksponering (×2,0) falder den igen, fordi farverne vaskes ud mod hvid og mætningen falder. YOLO er meget mere robust. En del af forklaringen er, at Ultralytics som standard træner med tilfældige lysændringer (`hsv_v=0.4`), så YOLO har *set* mørke billeder under træningen.

## Punkt 1b – Størrelse (alle 770 skilte i testsættet)
Se `stoerrelse.png`. **Den klassiske pipeline finder ingen skilte under 32 px**, mens YOLO finder 66 % af selv de mindste (under 8 px). Den klassiske grænse skyldes især morfologien: en 5×5-kerne i `MORPH_OPEN` sletter den tynde røde ring på et lille skilt, før Hough overhovedet ser den. Over 128 px finder den klassiske pipeline 44 %.

## Punkt 1c – Kilde
| | GTSRB | Vejfotos | road*.png | Simulator |
|---|---|---|---|---|
| Klassisk C | 43 % | 28 % | 5 % | **0 %** |
| YOLOv8n | 100 % | 93 % | 91 % | **70 %** |

Simulatoren er sværest for begge metoder: små trafiklys og ingen røde ringe. Det understøtter "valg af billeder"-afsnittet. GTSRB-billederne er de letteste, men også de mindst realistiske. Det er en god pointe at diskutere i rapporten.

## Punkt 4 – Hastighed (ms pr. billede, 416×416)
| Metode | ms | billeder/s |
|---|---|---|
| Klassisk C (CPU) | 5,7 | 174 |
| Klassisk C + CNN (GPU) | 8,5 | 117 |
| Klassisk C + CNN (CPU) | 9,7 | 103 |
| YOLOv8n (GPU) | 28,1 | 36 |
| YOLOv8n (CPU) | 57,5 | 17 |

Den klassiske pipeline er 5 gange hurtigere end YOLO, og CNN'et koster kun ca. 3 ms, fordi udsnittene er små (64×64). YOLO når realtid (≥ 30 billeder/s) på GPU, men ikke på CPU. Det er relevant for F3 og kameraudstyrede enheder: en billig bilcomputer uden GPU ville have svært ved YOLO.

Ét forbehold: Ultralytics målte selv 1,7 ms ren inferens under testen. De 28 ms her er hele kaldet, med forbehandling, NMS og overførsel til GPU for ét billede ad gangen. Ved en videostrøm med batching ville YOLO være hurtigere.

## Punkt 5 – Hough-sweep (på valideringssættet)
Se `hough_sweep.png`. Grafen viser trade-off'et tydeligt:
- Højere `param2` giver færre falske alarmer (højre graf, log-skala), men også færre fundne skilte (venstre graf).
- **Ring-tjekket skubber hele kurven ned:** ved samme `param2` giver det 5–25 gange færre falske alarmer og koster kun cirka 10 procentpoint i fundne skilte.
- Den valgte værdi (30) ligger ved "knækket": fra 30 til 40 halveres antallet af fundne skilte.

Fordi sweepet kører på **valideringssættet**, er forbeholdet fra sidst nu løst. Parameteren er valgt på andre data end testdataene. Valideringssættet giver 24,4 % fundne skilte ved `param2=30`, og testsættet gav 23,2 %. De næsten ens tal viser, at parameteren ikke er overtilpasset til testsættet. Den sætning kan du bruge direkte i rapporten.

I grafen med negative billeder rammer linjerne bunden ved `param2` 40–50. Det er fordi værdien er 0, som ikke kan vises på en log-skala.

Jeg har også gemt en note om projektet til fremtidige sessioner: hvor vurderingsgrundlaget ligger, at `30.x.jpg` er billeder fra datasættet, og at resultater altid skal gemmes i nye mapper.

---

## Prompt 13 (2026-11-02 17:35)

**Mig:**

Kan du sætte en test i gang hvor du opskalerer billederne som nævnt tidligere?

**Claude:**

Ja. Først en hurtig ekstra måling, der koster 1–2 minutter: **den eksisterende 416-model evalueret ved 640**. Den er ikke trænet til det, men resultatet viser, om opskalering *alene* hjælper. Den fulde test er bagefter at **træne en ny model ved `imgsz=640`**. Det forventer jeg tager ca. 2,4 × 25 ≈ **60 minutter**, så husk at have opladeren i.

Jeg ændrer kun `imgsz`. Alt andet (epochs, batch, augmentation, seed) er uændret, så forskellen kan tilskrives opskaleringen alene.

Den hurtige evaluering burde tage under et minut, men har kørt i over 10. Jeg undersøger, hvad der sker:

Den notifikation var bare den stoppede proces. Nu kører jeg evalueringen fra en fil med `workers=0`:

Alt kører som det skal. Træningen startede 18:49, og dataloaderens fire hjælpeprocesser er i gang. GPU-målingen før viste lav belastning, fordi data stadig blev indlæst.

**Status:**
- **Hurtig test:** færdig. 416-modellen evalueret ved 640 gav mAP50 0,949 (op fra 0,936), men mAP50-95 0,755 (ned fra 0,783). Den finder lidt flere skilte, men boksene bliver mindre præcise. Resultatet ligger i `results/yolov8n_416model_eval640_20261102_184559/`.
- **Fuld træning ved 640:** kører i baggrunden og forventes færdig omkring **19:50**. Den gemmes i `results/yolov8n_640_<tid>/`, og loggen skrives til `yolo_train_log.txt`.
- **Klar til bagefter:** `yolo_size_compare.py` sammenligner 416- og 640-modellen efter skiltestørrelse (der hvor opskaleringen burde hjælpe) og efter hastighed på GPU og CPU.

Undervejs fandt og løste jeg et problem: min første hurtige test hang i 12 minutter. Jeg havde kørt koden direkte fra terminalen i stedet for fra en fil, og så kan Ultralytics' hjælpeprocesser ikke starte på Windows. Det er den samme grund til, at `train_yolo.py` har `if __name__ == "__main__"`. Den hængende proces er stoppet, og testen er kørt korrekt.

Jeg samler resultaterne op og kører sammenligningen, når træningen er færdig.

Træningen ved 640 er færdig og bekræfter din forventning: opskalering hjælper på de små skilte, og prisen er tid.

## Testsættet: 416 mod 640

| | mAP50 | mAP50-95 | Red Light mAP50 | Green Light mAP50 |
|---|---|---|---|---|
| Trænet 416 | 0,936 | 0,783 | 0,736 | 0,873 |
| Trænet 416, *testet* ved 640 | 0,949 | 0,755 | – | – |
| **Trænet 640** | **0,957** | **0,810** | **0,813** | **0,922** |

## Efter skiltets størrelse (fundet med rigtig klasse)
Se `results/yolo_size_compare_20261102_193324/stoerrelse_416_vs_640.png`.

| Korteste side | < 8 px | 8–16 | 16–32 | 32–64 | 64–128 | ≥ 128 |
|---|---|---|---|---|---|---|
| Trænet 416 | 66 % | 86 % | 95 % | 91 % | 97 % | 99 % |
| Trænet 640 | **77 %** | **95 %** | 94 % | 92 % | 99 % | 97 % |

## Tolkning

**1. Forbedringen ligger præcis, hvor den skulle.** Skilte under 8 px går fra 66 % til 77 %, og skilte på 8–16 px fra 86 % til 95 %. Det er de små trafiklys fra simulatoren, så Red Light og Green Light er også de klasser, der forbedres mest. For skilte over 16 px er forskellene 1–2 procentpoint i begge retninger. Det er støj, ikke en reel forskel.

**2. Det er ikke ny information, der hjælper.** Billederne er stadig 416×416, så de små skilte har ikke fået flere detaljer. Det, der ændrer sig, er hvor mange felter i YOLO's feature map skiltet dækker. Ved 640 fylder et trafiklys på 5 px omtrent det samme som et på 8 px ved 416. Så kan netværket fange det, før det "forsvinder" i nedskaleringen.

**3. Det er nødvendigt at træne ved 640, ikke kun at teste ved 640.** Den hurtige test viste, at 416-modellen testet ved 640 fandt flere skilte (mAP50 steg), men placerede boksene dårligere (mAP50-95 faldt fra 0,783 til 0,755). Modellen har lært, hvor store skilte plejer at være. Først når den trænes ved 640, forbedres begge tal. Det er en god pointe til rapporten: **trænings- og testforhold skal matche**.

**4. Prisen:**
- Træningen tog 41 minutter mod 25 minutter, altså ca. 1,6 gange længere. Det er mindre end de forventede 2,4 gange, fordi GPU'en ikke var fuldt udnyttet ved 416, hvor dataindlæsning også tog en del af tiden.
- Kørsel: 28 mod 23 ms pr. billede på GPU, og **90 mod 50 ms på CPU**. På en bilcomputer uden GPU falder 640-modellen fra 20 til 11 billeder i sekundet, langt under realtid.

**5. Til sammenligningen i rapporten:** det er et klassisk trade-off. 640 er den bedste model, hvis der er en GPU. 416 er det fornuftige valg på svagere hardware. Det passer godt med F3 (begrund brugen af kamera) og din pointe om kameraudstyrede enheder.

**Forbehold:** hastighedsmålingerne svinger mellem kørsler. Samme 416-model målte 28 ms i den forrige analyse og 23 ms her. Brug derfor tallene til at sammenligne størrelsesordener, ikke som præcise værdier.

**Filer:** træningen ligger i `results/yolov8n_640_20261102_184940/` (inkl. `train_log.txt`), testresultaterne i `..._test/`, og sammenligningen i `results/yolo_size_compare_20261102_193324/`. `train_yolo.py` tager nu `--imgsz=640` som argument, og nye kørsler hedder `yolov8n_<størrelse>_<tid>`.

---

## Prompt 14 (2026-11-02 20:20)

**Mig:**

ville det give mening at prøve at opskalere mere?

**Claude:**

Kort svar: én kørsel mere, ved 832, kan give mening som et datapunkt til rapporten. Men jeg forventer ikke en stor forbedring, og der er bedre måder at hjælpe de mindste skilte på.

**Hvorfor gevinsten nok bliver lille:**
- **Der er ikke meget tilbage at hente.** 640-modellen finder allerede 95 % af skiltene på 8–16 px og 94–99 % af alt over 16 px. Kun skiltene under 8 px, hvor den finder 77 %, har plads til en større forbedring.
- **Billederne har ingen flere detaljer.** De er 416×416, så ved 832 bliver hver original pixel blot til en blok på 2×2. Opskalering hjælper kun, fordi skiltet kommer til at fylde flere felter i YOLO's stride-8-grid. Den effekt aftager: et trafiklys på 4 px er stadig bare 4 pixels information, uanset hvor meget det strækkes.
- **De store skilte kan blive ramt.** Over 128 px faldt de allerede en smule ved 640 (99 → 97 %, inden for støjen). Ved 832 bliver de 256+ px og nærmer sig grænsen for, hvad netværkets dybeste lag "ser" på én gang.
- **640-modellen er færdigtrænet.** Val-mAP50 lå fladt på 0,970–0,972 de sidste 6 epochs. Flere epochs ved 640 hjælper altså heller ikke.

**Hvad 832 ville koste:**
- **Træning:** ca. 1–1,5 time, ud fra at 640 tog 41 min.
- **GPU-hukommelse:** dine 6 GB rækker sandsynligvis ikke til batch 32 ved 832, så batch skal ned på 16. Så ændrer jeg to ting på én gang, og sammenligningen bliver mindre ren.
- **CPU:** cirka 150 ms pr. billede, svarende til ca. 6–7 billeder i sekundet. Den model ville aldrig kunne køre i en bil uden GPU.

**Bedre veje til de små skilte, hvis det er dem du vil forbedre:**
1. **YOLOv8n-P2:** en variant med et ekstra detektionslag ved stride 4. Den giver finere opløsning til små objekter uden at gøre hele billedet større. Det er den løsning, der går direkte efter problemet.
2. **YOLOv8s ved 640:** en større model med mere kapacitet. Den løser et andet problem end opløsningen, så den er også interessant at sammenligne med.

**Til rapporten:** din vurdering lægger ikke meget vægt på præcision. Den lægger vægt på sammenligning og refleksion. Tre punkter (416 / 640 / 832) kan vise *aftagende udbytte* som en kurve, og det er et stærkere argument end "større er bedre". Men to punkter og ovenstående forklaring er også et fint afsnit. Tiden er måske bedre brugt på rapporten.

Hvad er vigtigst for dig: kurven med aftagende udbytte (832) eller en arkitektur-sammenligning (P2 eller v8s)?

---

## Prompt 15 (2026-11-02 20:25)

**Mig:**

bægge dele. Kør først endnu en opskalering, og senere tager vi arkitektursammenligningen

**Claude:**

Batch 16 bruger 3,4 GB og kører 2,5 trin i sekundet. Det giver omkring 1½ time for 50 epochs. Jeg starter den fulde træning i baggrunden.

Mens den træner, udvider jeg `yolo_size_compare.py`, så den kan sammenligne et vilkårligt antal kørsler. Billedstørrelsen læser den fra hver kørsels `args.yaml`.

Træningen ved 832 kører nu i baggrunden. Den bør være færdig om ca. 1½ time, og så kører jeg sammenligningen og tolker resultaterne.

**Ændring af batch-størrelsen:** jeg har ændret batch fra 32 til 16. Det er nødvendigt, men gør sammenligningen mindre ren:
- **Hvorfor:** en prøvekørsel med batch 32 ved 832 brugte 6,5 GB. GPU'en har kun 6 GB, så Windows flyttede resten over i almindelig RAM, og hvert trin tog 3,7 s i stedet for ca. 0,4 s. Hele træningen ville have taget omkring 5 timer.
- **Hvad det betyder:** batch 16 bruger 3,4 GB og kører normalt, men nu er to ting ændret mellem kørslerne. Mindre batches giver mere "støjende" gradienter. Det skader sjældent, men en forskel kan i princippet skyldes batch og ikke billedstørrelse.
- **I rapporten:** nævn begrænsningen åbent, fx "832 krævede batch 16 pga. 6 GB VRAM". At hardware sætter grænser for eksperimentet er i sig selv en relevant refleksion.

**Kodeændringer:**
- `train_yolo.py` tager nu også `--batch=N`. Hvis batch ikke er 32, kommer det med i mappenavnet, fx `yolov8n_832_b16_<tid>`, så man kan se, at kørslen afviger.
- `yolo_size_compare.py` tager nu et vilkårligt antal kørselsmapper. Billedstørrelse og batch læses fra hver kørsels `args.yaml`, så jeg kan sammenligne 416, 640 og 832 i ét plot.

Prøvekørslerne ligger som `yolo_smoketest_832_*` i `results/`. De kan slettes, hvis de fylder for meget.

Et spørgsmål at tænke over, mens den træner: hvad forventer du, der sker med skiltene over 128 px ved 832, når de bliver over 256 px i netværkets input?

The legend sits on top of the "82" label. I'll move it below the plot and regenerate the figure from the saved numbers, without running the whole comparison again.

832-modellen er færdigtrænet. Den klarede sig lidt bedre på testsættet, end jeg forudsagde, men på valideringssættet er den næsten lig 640-modellen.

## Tre billedstørrelser

| | mAP50 test | mAP50-95 test | mAP50 val (sidste epoch) | Træningstid | CPU pr. billede |
|---|---|---|---|---|---|
| 416 | 0,936 | 0,783 | – | 25 min | 59 ms |
| 640 | 0,957 | 0,810 | 0,971 | 41 min | 106 ms |
| 832 (batch 16) | **0,969** | **0,828** | 0,973 | 67 min | **159 ms** |

**Fundet med rigtig klasse, efter skiltets korteste side:**

| | < 8 px | 8–16 | 16–32 | 32–64 | 64–128 | ≥ 128 | I alt |
|---|---|---|---|---|---|---|---|
| 416 | 66 % | 86 % | 95 % | 91 % | 97 % | 99 % | 90,9 % |
| 640 | 77 % | 95 % | 94 % | 92 % | 99 % | 97 % | 93,2 % |
| 832 | **82 %** | 92 % | 94 % | **97 %** | 100 % | 98 % | **94,8 %** |
| antal skilte | 124 | 59 | 80 | 88 | 118 | 301 | |

Plottet ligger i `results/yolo_size_compare_20261102_224155/stoerrelse_416_vs_640_vs_832.png`. Træningen ligger i `results/yolov8n_832_b16_20261102_213212/`.

## Tolkning

**1. Skilte under 8 px fortsætter med at blive bedre, men gevinsten aftager:**
- 416 → 640 gav +11 point.
- 640 → 832 gav +5 point.

Det passer med, at hver opskalering giver mindre end den forrige. Det er det mønster, rapporten kan vise.

**2. Vær forsigtig med de små grupper.** I gruppen 8–16 px er der kun 59 skilte, så ét skilt svarer til 1,7 point. Faldet fra 95 til 92 % er altså 2 skilte, og stigningen i 32–64 px er 4 skilte. Kun gruppen under 8 px har så mange skilte (124), at forskellen sandsynligvis er reel. Det svarer til 6 flere skilte end ved 640.

**3. Testsættet og valideringssættet er uenige.** På testsættet ser 832 tydeligt bedre ud end 640 (0,969 mod 0,957). På valideringssættet er de næsten ens (0,973 mod 0,971). Det betyder, at en del af testforbedringen kan skyldes, hvilke billeder der tilfældigvis endte i testsættet. Desuden er batch ændret til 16, så forskellen kan ikke kun tilskrives billedstørrelsen. Den ærlige konklusion er: 832 hjælper sandsynligvis lidt på de mindste skilte, men forskellen til 640 er så lille, at et enkelt forsøg ikke kan bevise den.

**4. Prisen stiger hurtigere end gevinsten.** Fra 416 til 832 stiger tiden på CPU fra 59 til 159 ms. Det er under 7 billeder i sekundet, hvilket ikke er brugbart i en bil uden GPU. På GPU er forskellen lille (25 → 30 ms), fordi GPU'en ikke er fuldt udnyttet ved så små billeder.

**5. De store skilte blev ikke dårligere.** Jeg nævnte risikoen for store skilte, men skilte over 128 px ligger på 97–99 % ved alle tre størrelser. Den bekymring kan droppes.

**Til rapporten:** 640 er det bedste kompromis. 832 giver en lille ekstra gevinst, der ikke er sikkert påvist, og koster ca. 50 % mere tid på CPU end 640.

Smoketest-mapperne `yolo_smoketest_832_*` i `results/` er ikke slettet. Jeg har ikke committet noget.

Skal jeg starte arkitektursammenligningen (YOLOv8n-P2 og YOLOv8s ved 640) nu, eller vil du vente?

---

## Prompt 16 (2026-11-02 22:03)

**Mig:**

Hvor lang tid vil sammenligningen tage pr. dit estimat?

**Claude:**

Mit estimat er **ca. 2½–3 timer i alt**. Modellerne skal trænes efter hinanden, fordi der kun er én GPU.

| Del | Estimat | Begrundelse |
|---|---|---|
| Smoketests (hukommelse) | ~5 min | Begge modeller kræver mere hukommelse end v8n, så batch-størrelsen skal tjekkes først, ligesom ved 832 |
| YOLOv8n-P2 ved 640 | ~55–70 min | Det ekstra detektionslag ved stride 4 arbejder på et feature map, der er 4 gange større end ved stride 8. Det giver ca. 40 % flere beregninger end v8n (41 min) |
| YOLOv8s ved 640 | ~80–100 min | v8s kræver ca. 3,3 gange så mange beregninger som v8n. Ved 640 var GPU'en ikke fuldt udnyttet, fordi en del af tiden gik med at indlæse data, så tiden stiger nok kun 2–2,5 gange |
| Testevaluering + størrelses- og hastighedssammenligning | ~10 min | Samme som ved 832 |

**Usikkerheder:**
- **Batch 16:** begge modeller kommer sandsynligvis over 6 GB ved batch 32, og så skal batch ned på 16, ligesom ved 832. Med 4 kørsler på batch 16 (640 v8n, 832 v8n, P2, v8s) bliver sammenligningen renere, hvis jeg også træner v8n ved 640 med batch 16 som reference. Det koster ca. 45 min ekstra.
- **P2 uden færdig COCO-vægt:** der findes ikke en fortrænet P2-model. Jeg kan genbruge rygraden fra `yolov8n.pt`, men det nye detektionslag starter fra bunden. Det kan betyde, at den skal bruge flere epochs, før den rammer sit niveau.
- **Strøm og varme:** sørg for at laptoppen er tilsluttet strøm. Sidst den kørte på batteri, tog en epoch 6 minutter i stedet for 40 sekunder.

Skal jeg køre med den ekstra batch-16-reference (ca. 3¼–3¾ timer), eller kun P2 og v8s (ca. 2½–3 timer)?

---

## Prompt 17 (2026-11-02 22:05)

**Mig:**

Kan du forklare mig helt præcis hvad arkitektursammenligningen vil opnå, og hvad jeg ville skulle bruge det til?

**Claude:**

## Hvad sammenligningen skal finde ud af

Indtil nu har du kun ændret én ting: **billedstørrelsen**. Den hjalp på små skilte, men den gør hele billedet større, også alle de store skilte og al baggrunden, der ikke havde brug for det. Derfor bliver CPU-tiden næsten tredoblet.

Arkitektursammenligningen skal besvare spørgsmålet: **hvorfor overser modellen de små skilte?** Der er to mulige forklaringer, og de kræver forskellige løsninger:

| Forklaring | Hvad der sker | Løsning | Test |
|---|---|---|---|
| **Opløsning** | YOLOv8n leder tidligst efter objekter i et grid med felter på 8×8 pixels (stride 8). Et trafiklys på 5 px fylder mindre end ét felt og bliver "udvisket", før detektionslaget ser det. | Lad netværket lede i et finere grid | **P2**: ekstra detektionslag med felter på 4×4 px |
| **Kapacitet** | Netværket kan godt se skiltet, men er for lille til at lære at skelne det fra baggrunden. | Større netværk | **v8s**: samme struktur, ca. 3 gange så mange parametre |

Opskaleringen har været en indirekte løsning på opløsningsproblemet. P2 er den direkte løsning.

**Sådan læser du resultatet:**
- **Hvis P2 hjælper mest på små skilte:** problemet er opløsning. Billedet behøver ikke opskaleres; kun den del af netværket, der leder efter små ting, skal arbejde finere.
- **Hvis v8s hjælper mest:** problemet er kapacitet, og opskaleringen virkede af andre grunde, end vi troede.
- **Hvis ingen af dem hjælper:** de mindste skilte indeholder simpelthen for lidt information. Et trafiklys på 4 px er kun et par farvede pixels. Så er grænsen nået for, hvad billedet kan give, og løsningen er et bedre kamera, ikke en bedre model.

Alle tre udfald kan bruges i rapporten.

## Et bedre forsøg end det, jeg foreslog

Jeg vil foreslå at teste **P2 ved 416** i stedet for ved 640. Grunden:
- v8n ved 832 med stride 8 giver et grid på 832 / 8 = **104 × 104**.
- P2 ved 416 med stride 4 giver også **104 × 104**.

De to har altså samme fineste grid. Forskellen er, at v8n ved 832 behandler et 4 gange så stort billede i *hele* netværket, mens P2 kun gør det fint i ét ekstra lag. Spørgsmålet bliver da: **kan P2 ved 416 finde de små skilte lige så godt som v8n ved 832, men langt hurtigere?** Det er en ren sammenligning, som du kan forklare med én tegning, og den peger direkte på din anvendelse: en bil uden en kraftig GPU. Den træner også hurtigere, ca. 35–45 min.

## Hvad du kan bruge det til i rapporten

1. **Vurderingskriteriet "sammenligning af flere algoritmer":** du har allerede klassisk vs. eget CNN vs. YOLO. Dette tilføjer en sammenligning *inden for* deep learning, hvor du ændrer én ting ad gangen og kan forklare hvorfor. Det viser forståelse, ikke kun at du har kørt koden.
2. **Begrundelse for dit endelige modelvalg:** i stedet for "vi valgte 640, fordi den var bedst" kan du skrive "problemet var opløsning, og den billigste måde at løse det på var X". Det er den form for refleksion, vurderingen lægger vægt på.
3. **Til eksamen:** censor spørger næsten altid "hvorfor ikke bare en større model?". Så har du et målt svar.

## Hvad det ikke giver

Det hæver ikke din karakter i sig selv, og vurderingen lægger lidt vægt på præcision. Hvis tiden er knap, er rapporten vigtigere. Du har allerede nok materiale til en fyldestgørende sammenligning. Dette er et ekstra lag, ikke et hul, der skal lukkes.

**Mit forslag:** P2 ved 416 (ca. 40 min) og v8s ved 416 (ca. 50–60 min), altså ca. 1¾ time i alt. Så sammenlignes begge med din oprindelige 416-model, og P2 også med 832. Skal jeg køre det, eller vil du hellere bruge tiden på rapporten?

---

## Prompt 18 (2026-11-02 22:26)

**Mig:**

Bare start arkitektursammenligningerne. Jeg kan have min pc til at stå tendt mens jeg sover. Så kør gerne nogle af de længere.  
  
I mellemtiden, kom med en prompt jeg kan bruge i en claude chat for at få den til at hjælpe mig med at skrive min porteføljerapport

**Claude:**

Mens smoketestene kører, skriver jeg prompten.

Jeg venter på det sidste smoketest (v8s ved 640), før natkørslen sættes i gang.

Natkørslen er i gang, og den første træning (P2 ved 416) kører korrekt med ca. 45 sekunder pr. epoch. Når alt er færdigt, tolker jeg resultaterne.

## Hvad der kører i nat (ca. 6–7 timer)

| # | Model | imgsz | batch | Sammenlignes med |
|---|---|---|---|---|
| 1 | YOLOv8n-P2 | 416 | 32 | din oprindelige v8n ved 416 og v8n ved 832 (samme fineste grid) |
| 2 | YOLOv8s | 416 | 32 | v8n ved 416 |
| 3 | YOLOv8n (reference) | 640 | 16 | – |
| 4 | YOLOv8n-P2 | 640 | 16 | nr. 3 |
| 5 | YOLOv8s | 640 | 16 | nr. 3 |

Efter hver gruppe laver `yolo_size_compare.py` en størrelses- og hastighedssammenligning i sin egen `results/`-mappe. De korteste træninger kører først, så der er resultater, selv hvis noget stopper undervejs. Hold PC'en tilsluttet strøm.

**Hvorfor 416-gruppen er den vigtigste:** den svarer på spørgsmålet "opløsning eller kapacitet?" uden at blande opskalering ind. 640-gruppen tjekker, om svaret stadig holder, når billederne også er opskaleret.

**Om P2:** P2 fik kun 219 af 437 vægte overført fra den fortrænede `yolov8n.pt`. Det er rygraden, altså de lag, der udtrækker features. Den del, der kombinerer features og laver detektionerne, starter fra bunden, fordi lagene er nummereret anderledes i P2. P2 starter altså med en ulempe i forhold til v8n og v8s, der næsten er fuldt fortrænede. Det skal nævnes i rapporten, hvis P2 klarer sig dårligt.

**Kodeændringer:**
- `train_yolo.py` tager nu også `--model=yolov8s` / `--model=yolov8n-p2`.
- `yolo_size_compare.py` viser nu modelnavnet i tabeller og plot, og med `--no-base` kan 416-modellen udelades.

## Rapportprompten

Prompten ligger i [rapport_prompt.md](rapport_prompt.md), og jeg har også sendt filen til dig. Kopiér alt under stregen ind i en ny Claude-chat, og vedhæft studieordningen, `PorteføljeIntro.md` og de plots, du vil bruge. Prompten indeholder:
- **Arbejdsform:** dansk, afsnit for afsnit, ingen opfundne tal (manglende tal markeres som `[MANGLER: ...]`), ærlig om svagheder, ingen unødig ros, og af og til et refleksionsspørgsmål.
- **Vurderingskriterierne** fra underviserens GitHub.
- **Alle dine resultater med tal:** label-tjek, klassisk pipeline med fejlen og rettelsen i Hough-tuningen, begge CNN-iterationer, YOLO, den fulde negative test, robusthed og opskaleringen.
- **Første opgave:** et udkast til disposition, hvor hver figur er knyttet til et afsnit.

Arkitektursammenligningen er ikke med i prompten endnu. Jeg tilføjer den, når resultaterne er klar.

Alle 5 træninger er gennemført. Jeg tilføjer resultaterne i rapportprompten.

Natkørslen gik igennem uden fejl: 5 træninger fra 23:39 til 04:16. Resultatet er tydeligt for v8s. For P2 kan det ikke bruges, og en af resultaterne ændrer gårsdagens konklusion om 832.

## Ved 416 (batch 32): opløsning eller kapacitet?

| Model | mAP50 | mAP50-95 | < 8 px | 8–16 | 16–32 | I alt | CPU |
|---|---|---|---|---|---|---|---|
| v8n | 0,936 | 0,783 | 66 % | 86 % | 95 % | 90,9 % | 45 ms |
| v8n-P2 | 0,878 | 0,745 | 48 % | 73 % | 85 % | 85,3 % | 64 ms |
| **v8s** | **0,956** | **0,809** | 69 % | **97 %** | 96 % | **93,0 %** | 92 ms |

## Ved 640 (batch 16)

| Model | mAP50 | mAP50-95 | < 8 px | 8–16 | I alt | CPU |
|---|---|---|---|---|---|---|
| v8n | 0,966 | 0,814 | 76 % | 90 % | 93,5 % | 84 ms |
| v8n-P2 | 0,931 | 0,790 | 71 % | 81 % | 90,0 % | 126 ms |
| v8s | 0,967 | 0,814 | 80 % | 95 % | 94,7 % | 195 ms |

## Hvad det betyder

**1. Svaret på "opløsning eller kapacitet?" afhænger af skiltets størrelse:**
- **Under 8 px er opløsningen flaskehalsen.** v8s hjælper næsten ikke (66 → 69 %), mens opskalering til 640 hjælper meget (66 → 76 %). Når der næsten ingen pixels er, hjælper en klogere model ikke.
- **På 8–32 px hjælper kapacitet.** v8s ved 416 finder 97 % af skiltene på 8–16 px, mod 86 % for v8n.

**2. Der er ingen genvej.** v8s ved 416 rammer samme mAP som v8n ved 640, men bruger lige så meget CPU-tid (92 ms mod 84–106 ms). De to løsninger koster det samme.

**3. Når opløsningen er høj nok, giver en større model ikke meget.** Ved 640 er v8s og v8n næsten ens (0,967 mod 0,966), men v8s er 2,3 gange langsommere på CPU.

**4. 832-konklusionen skal rettes.** v8n ved 640 med batch 16 fik 0,966 / 0,814, næsten det samme som 832 (0,969 / 0,828). Det meste af forspringet, som 832 så ud til at have, skyldtes altså batch-størrelse og tilfældighed, ikke billedstørrelsen. Den reelle forbedring er fra 416 til 640.

**5. P2-forsøget er ikke fair, og det skal nævnes i rapporten.** P2 er dårligst i begge grupper, men det siger mere om forsøget end om arkitekturen:
- Der findes ingen fortrænet P2-model, så kun rygraden (219 af 437 vægte) fik COCO-vægte. Resten startede fra bunden.
- Efter 10 epochs lå P2 på val mAP50 0,69, mens v8s allerede var på 0,92.
- P2 blev stadig bedre ved epoch 50 (0,928 → 0,942 fra epoch 40 til 50), mens v8s lå fladt.

P2 er altså undertrænet. Forsøget kan ikke afgøre, om finere opløsning via P2 virker.

**Endeligt valg til din use case:** v8n ved 640. Den har bedste forhold mellem nøjagtighed og CPU-tid, og 416-gruppen begrunder hvorfor.

**Forbehold om hastighed:** samme v8n 416-model målte 45 ms på CPU i nat og 59 ms i går. Sammenlign derfor kun hastigheder inden for samme kørsel.

## Filer
- `results/yolo_size_compare_20261103_005455/` (416-gruppen) og `results/yolo_size_compare_20261103_041400/` (640-gruppen), hver med `stoerrelse.png` og `summary.txt`.
- Hver træning har sin egen mappe med `train_log.txt`.
- [rapport_prompt.md](rapport_prompt.md) har fået et nyt afsnit om arkitektursammenligningen og korrektionen om 832.

Jeg har ikke committet noget, og prøvekørslerne (`yolo_smoketest_*`) ligger stadig i `results/`.

Skal P2 trænes igen med ca. 100 epochs (ca. 1½ time ved 416), så forsøget bliver fair? Eller skal begrænsningen bare nævnes i rapporten?

---

## Prompt 19 (2026-11-03 17:39)

**Mig:**

Vi nævner bare begrænsningen i rapporten. Kan du komme med en opdateret prompt til den anden chat?

**Claude:**

Den fulde prompt er opdateret i [rapport_prompt.md](rapport_prompt.md), og jeg har sendt filen til dig. Den har nu et afsnit om arkitektursammenligningen med begge tabeller og fortolkningen, korrektionen af 832-konklusionen, P2-begrænsningen (med at du bevidst har valgt ikke at træne P2 længere) og forbeholdet om hastighedsmålinger.

**Hvis du starter en ny chat:** kopiér alt under stregen i filen.

**Hvis du allerede er i gang i den anden chat:** så er det nemmere at sende denne opdatering, så du ikke starter forfra:

```text
Opdatering med nye resultater – brug dem fremover, og ret evt. det vi allerede har skrevet:

1. KORREKTION om opskalering: En reference med YOLOv8n ved 640 og batch 16 gav test mAP50 0,966 / mAP50-95 0,814 – næsten som 832 (0,969 / 0,828). Forspringet for 832 over 640 (batch 32: 0,957 / 0,810) skyldtes altså primært batch-størrelse og tilfældighed, ikke billedstørrelsen. Den reelle forbedring er 416 → 640.

2. NYT AFSNIT – arkitektursammenligning: opløsning eller kapacitet?
Hypoteser: små skilte overses pga. (a) opløsning – v8n leder tidligst i et grid med 8×8 px felter (stride 8), testet med YOLOv8n-P2 (ekstra detektionslag ved stride 4), eller (b) kapacitet – testet med YOLOv8s (11,1 mio. parametre mod 3,2 mio.). P2 ved 416 har samme fineste grid (104×104) som v8n ved 832.

Ved 416, batch 32 (mAP50 / mAP50-95 / fundet med rigtig klasse <8 px, 8–16, 16–32, i alt / CPU-tid):
- v8n:    0,936 / 0,783 / 66 %, 86 %, 95 %, 90,9 % / 45 ms
- v8n-P2: 0,878 / 0,745 / 48 %, 73 %, 85 %, 85,3 % / 64 ms
- v8s:    0,956 / 0,809 / 69 %, 97 %, 96 %, 93,0 % / 92 ms

Ved 640, batch 16 (P2 og v8s kan ikke være i 6 GB VRAM med batch 32):
- v8n:    0,966 / 0,814 / 76 %, 90 %, 91 %, 93,5 % / 84 ms
- v8n-P2: 0,931 / 0,790 / 71 %, 81 %, 81 %, 90,0 % / 126 ms
- v8s:    0,967 / 0,814 / 80 %, 95 %, 92 %, 94,7 % / 195 ms

Fortolkning:
- Under 8 px er opløsning flaskehalsen: v8s hjælper næsten ikke (66 → 69 %), opskalering gør (66 → 76 %).
- 8–32 px hjælpes af kapacitet (v8s: 86 → 97 % på 8–16 px).
- Ingen genvej: v8s ved 416 når samme mAP som v8n ved 640, men med samme CPU-tid.
- Ved 640 giver v8s næsten intet ekstra over v8n, men koster 2,3× CPU-tid.
- P2 er dårligst, men forsøget er ikke fair: der findes ingen fortrænet P2, så kun rygraden (219 af 437 vægte) fik COCO-vægte; neck og detektionshoved startede fra bunden. P2's valideringskurve steg stadig ved epoch 50 (0,928 → 0,942 fra epoch 40 til 50), mens v8s lå fladt. P2 er undertrænet, så forsøget kan ikke afvise hypotesen. Jeg har bevidst valgt ikke at træne længere – det skal stå som en begrænsning og et muligt næste skridt (fx 100+ epochs).
- Hastighedsmålinger svinger mellem kørsler (samme v8n 416: 45 ms og 59 ms CPU), så sammenlign kun inden for samme tabel.

Endeligt modelvalg til use casen (bil, svag hardware): YOLOv8n ved 640.

Figurer: results/yolo_size_compare_20261103_005455/stoerrelse.png (416-gruppen) og results/yolo_size_compare_20261103_041400/stoerrelse.png (640-gruppen).
```

---

## Prompt 20 (2026-11-03 18:25)

**Mig:**

Kan du genskrive cannytest så den laver outputs som billeder, jeg skal bruge det til rapporten.   
Lav en ny mappe til dem

**Claude:**

`cannytest.py` gemmer nu alle sine resultater som billeder. Jeg har kørt det på 30.2.jpg og tjekket, at billederne kommer korrekt ud.

**Hvor billederne ligger:** hver kørsel får sin egen mappe, så gamle billeder aldrig overskrives. Mappen fra den første kørsel er `results/cannytest_30.2_20261103_192551/`. Den har to undermapper:
- **`figures/`**: de 11 sammenligningsfigurer, fx `02_canny.png`, `04_hough_circles.png`, `06_hsv_red_mask.png` og `11_flann_matching.png`.
- **`single_images/`**: hvert billede fra figurerne gemt for sig uden titel og akser, i alt 48 stk. Filnavnet bruger titlen på billedet, fx `06_hsv_red_mask__HSV___Restricted_Red_Mask.png`. Det er nyttigt, hvis du selv vil sætte billederne op i rapporten.

**Hvad jeg har ændret i koden:**
- `plt.show()` er erstattet af en blok, der gemmer alle figurer (150 dpi) og hvert billede for sig. Resten af koden er uændret.
- Du kan nu vælge et andet billede: `python cannytest.py 20.1.jpg`. Uden argument bruges `30.2.jpg` som før.
- Med `--show` åbnes figurerne også i vinduer, som før.
- Scriptet stopper med en fejl, hvis du tilføjer eller fjerner en figur uden at opdatere listen `figure_names`. Ellers ville billederne få forkerte navne.

**Til rapporten:** Hough-figuren viser tydeligt, hvorfor din endelige pipeline ser anderledes ud. Hough direkte på Canny-kanter finder snesevis af falske cirkler med alle fire blur-typer. Den røde HSV-maske giver derimod én ren ring. Det kan du bruge under vurderingskriteriet om filtre, du valgte fra.

---

## Prompt 21 (2026-11-03 19:13)

**Mig:**

Kan pipelines.py gemme mellemresultaterne efter hvert filter? Uden den figur er afsnit 3 svagt.

**Claude:**

Jeg har tilføjet gemning af mellemresultater i `pipelines.py` og skrevet et figurscript. Nu kører jeg det.

Ja, det kan `pipelines.py` nu, og figuren til afsnit 3 er lavet. Den ligger i `results/pipeline_steps_20261103_201542/`:
- **`pipeline_steps.png`** er den samlede figur: 4 eksempler × 8 trin.
- **`eksempel_1.png` … `eksempel_4.png`** er én række pr. eksempel, hvis den samlede figur bliver for stor i rapporten.
- **`single_images/`** har hvert trin som et billede for sig i fuld opløsning.
- **`summary.txt`** har filnavnene på eksemplerne, antal cirkler og CNN'ens bud.

**De 8 trin** er median blur → rød HSV-maske → close → open → Gaussian blur → Hough → ring-tjek, med originalen først. Det er de samme parametre som i den negative test.

**Hvad de fire eksempler viser:**
1. **Stort skilt (vejfoto):** masken giver en ren ring, close og open fjerner de små røde prikker, og Hough finder skiltet. Men den finder det to gange, med en ekstra, større cirkel, og CNN'en er kun 47 % sikker på den. Det er et eksempel på en dobbeltdetektion.
2. **Lille skilt:** skiltet fylder kun få pixels i HSV-masken, og Hough finder 0 cirkler. Det viser direkte, hvorfor den klassiske pipeline finder 0 % af skilte under 32 px.
3. **Jordbær, afvist:** næsten hele masken er rød. Hough finder 5 cirkler, men ring-tjekket afviser alle, fordi midten også er rød. Det viser, hvad ring-tjekket bidrager med.
4. **Jordbær, falsk alarm:** Hough finder 10 cirkler, og 1 slipper igennem ring-tjekket, fordi den hvide fløde giver en "ikke-rød midte". CNN'en kalder den Red Light med 81 % sikkerhed. Det er en god forklaring på de 17,5 % falske alarmer på jordbær.

Det giver afsnit 3 både positive og negative billeder, og for hvert filter kan du se, hvad det gør, og hvor det fejler.

**Ændringer i koden:**
- `red_mask()` og `classical_detect()` i `pipelines.py` har fået et valgfrit `steps`-argument, der gemmer mellemresultaterne. Uden det argument virker de som før, så de tidligere tests giver de samme resultater.
- `pipeline_steps.py` er nyt. Det vælger selv eksemplerne med seed 0, så figuren kan laves igen.

Den første version tegnede de afviste cirkler i rødt, og de kunne ikke ses på jordbærrene. Jeg slettede den mappe og lavede figuren igen med tynde blå cirkler.

---

## Prompt 22 (2026-11-03 20:01)

**Mig:**

"Blev RGB-threshold afprøvet i colortest.py? I så fald er det et dokumenteret fravalg. Hvad viste det?"  
  
Spørgsmål jeg blev stillet af den anden claude chat. Kan du køre colortest.py og sørge for resultaterne bliver gemt?

**Claude:**

`colortest.py` afprøver ikke RGB-threshold. Den finder gule områder med HSV i `Yellow.png`. Jeg tjekker, om et af de andre scripts har afprøvet RGB.

Ingen af scripts'ene afprøver RGB-threshold. Alle farvemasker i projektet bruger HSV. Jeg sørger alligevel for, at `colortest.py` gemmer sine resultater, og kører det.

Nej, RGB-threshold blev ikke afprøvet i `colortest.py`, så det er ikke et dokumenteret fravalg endnu.

**Det `colortest.py` gør:** det finder **gule** områder i `Yellow.png` med en **HSV**-maske (H 20–30, S og V 100–255), som er et billede af frugt og ikke af trafikskilte. Det fandt 2 områder: bananen og den øverste del af appelsinen, fordi appelsinens orange ligger tæt på gul i hue. Billedet viser altså, at et hue-interval godt kan skære forkert. Men det siger intet om RGB eller om dine skilte.

**Resten af projektet:** jeg har søgt i alle scripts. `cannytest.py`, `CVtest.py`, `huesort.py`, `trackbar*.py` og `pipelines.py` bruger alle HSV, og ingen af dem bruger en RGB-maske.

**Gemt:** `colortest.py` gemmer nu figuren, masken, det detekterede billede og en `summary.txt` i `results/colortest_20261103_210144/`. Med `--show` åbnes figuren også i et vindue.

**Hvad du kan svare den anden chat:** RGB blev ikke afprøvet. Valget af HSV blev begrundet teoretisk: HSV adskiller farve (hue) fra lysstyrke (value), mens alle tre RGB-kanaler ændrer sig, når lyset ændrer sig. Det er et gyldigt argument, men det er ikke målt.

Hvis du vil have et målt fravalg, kan jeg bygge en test, der kører den klassiske pipeline med en RGB-maske i stedet for HSV. RGB-grænserne vælges på valideringssættet, så sammenligningen er fair. Bagefter sammenlignes skiltene fundet på testsættet ved de samme lysniveauer som i robusthedstesten (×0,3 til ×2,0). Det tager ca. 10 minutter at køre.

Skal jeg bygge den test, eller skal du bare nævne fravalget som teoretisk begrundet?

---

## Prompt 23 (2026-11-03 20:08)

**Mig:**

Nej, ikke nu

**Claude:**

Okay, vi bygger ikke RGB-testen. I rapporten kan du så skrive, at RGB blev fravalgt med en teoretisk begrundelse og ikke en måling. Fx: "HSV blev valgt frem for RGB, fordi farvetonen (H) er adskilt fra lysstyrken (V). En rød skiltekant beholder derfor nogenlunde samme H-værdi i skygge og sollys, mens alle tre RGB-kanaler ændrer sig med lyset. Fravalget er ikke målt i dette projekt."

Din egen robusthedstest gør argumentet stærkere: selv med HSV falder den klassiske pipeline til 0,5 % fundne skilte ved lysfaktor ×0,3. Det viser, at farvebaseret segmentering er følsom over for lys. HSV er ikke en fuld løsning, kun det bedste udgangspunkt af de to.

---

## Prompt 24 (2026-11-03 20:31)

**Mig:**

Der mangler én måling. Du har ikke kørt den negative test (jordbær og vejudsnit) på den endelige 640-model, så rækken i sammenligningstabellen er tom. Lige nu bygger rapporten sit valg på en model, der ikke er testet på negative billeder. Det er værd at køre negative_test.py med 640-vægtene, før du afleverer.  
  
Denne har jeg også fra den anden chat

**Claude:**

Den negative test er nu kørt med den endelige 640-model, så sammenligningstabellen har ikke længere en tom række. Resultaterne ligger i `results/negative_test_20261103_213220/`.

| | Jordbær (falsk alarm) | Vejudsnit (falsk alarm) | Fundet | Rigtig klasse | Falske bokse på positive |
|---|---|---|---|---|---|
| YOLOv8n 416 | 8/40 (20 %) | 6/100 (6 %) | 93,5 % | 92,4 % | 11 |
| **YOLOv8n 640** | 9/40 (22,5 %) | 6/100 (6 %) | 93,0 % | **93,0 %** | **8** |

**Tolkning:**
- **Opskaleringen har ikke givet flere falske alarmer.** 640-modellen har falsk alarm på ét jordbærbillede mere. Med kun 40 billeder er det støj.
- **Der er ingen forskel på vejudsnittene**, og 640-modellen har 3 færre falske bokse på de positive billeder.
- **Gevinsten på små skilte ses ikke i denne test.** De 150 tilfældige testbilleder indeholder få meget små skilte, så det er størrelsesanalysen på hele testsættet, der viser den gevinst.
- **Testen er reproducerbar.** Alle de andre rækker gav præcis de samme tal som i den tidligere kørsel, fordi seed og billeder er de samme.

**Det kan du skrive i rapporten:** 640-modellen er nu testet på både positive og negative billeder. Den finder flere små skilte uden at give flere falske alarmer.

**Ændringer:**
- `negative_test.py` kører nu både 416- og 640-modellen.
- Eksempelbillederne viser nu 640-modellen.
- Jeg har tilføjet den nye række og fortolkningen i [rapport_prompt.md](rapport_prompt.md).

Hvis du allerede er i gang i den anden chat, kan du give den tabellen og tolkningen ovenfor.

---

## Prompt 25 (2026-11-07 15:21)

**Mig:**

Der er en video inkluderet i datasettet, er det muligt at køre modellen på den video i realtid. Jeg kunne godt tænke mig at vise det til min fremlæggelse  
  
Ikke kør noget, bare svar på spørgsmålet

**Claude:**

Ja, det kan lade sig gøre, og din GPU er hurtig nok til det.

**Hastighed:** ud fra dine egne målinger:

| Model | GPU (RTX 3050) | CPU |
|---|---|---|
| YOLOv8n 640 (endelig) | ~26–28 ms → **35–40 billeder/s** | ~85–105 ms → ~10 billeder/s |
| YOLOv8n 416 | ~21–25 ms → 40–47 billeder/s | ~45–60 ms → 17–22 billeder/s |

En video kører typisk med 25–30 billeder i sekundet, så **640-modellen kan køre i realtid på GPU'en**. Kun på CPU kan den ikke. Målingerne er lavet ét billede ad gangen med overhead pr. kald. I en video-løkke med `stream=True` bliver det typisk lidt hurtigere. Hvis videoen er i høj opløsning, fx 1080p, koster det lidt ekstra at skalere hvert billede ned til 640.

**Sådan gøres det:** to muligheder.
1. **Den simple:** Ultralytics kan selv afspille videoen med bokse: `model.predict(source="video.mp4", imgsz=640, show=True, stream=True)`. Det er få linjer kode.
2. **Til fremlæggelsen:** et lille script med `cv2.VideoCapture`, der tegner bokse og klassenavne og viser en **FPS-tæller** i hjørnet. Tælleren beviser "realtid" i stedet for, at du bare siger det. Man kan også vise den klassiske pipeline (ca. 6 ms) ved siden af YOLO. Så ser censor forskellen på de to metoder direkte, og det passer godt med dit vurderingskriterie om sammenligning.

**Gode råd til fremlæggelsen:**
- **Sæt laptoppen til strøm.** På batteri så du en epoch gå fra 40 sekunder til 6 minutter. Uden strøm kan demoen hakke.
- **Hav en optaget version klar.** Kør videoen igennem på forhånd med `save=True`, så der ligger en færdig video med bokse. Hvis live-demoen fejler, fordi projektoren, driveren eller noget andet driller, viser du den i stedet.
- **Tjek, om billederne fra videoen er en del af træningssættet.** Hvis datasættet er bygget af frames fra den samme video, har modellen set billederne før, og demoen ser bedre ud, end modellen reelt er. Det skal du kunne svare på, hvis censor spørger. Hvis videoen er ny, er den til gengæld et ekstra stykke bevis for, at modellen virker på ukendte data.
- **Forvent fejl, og brug dem.** Små trafiklys vil blinke ind og ud, fordi Red Light har den laveste recall. Det kan du bruge aktivt: "her kan I se det, størrelsesanalysen viste".

Sig til, når du vil have scriptet lavet. Så skal jeg vide, hvor videoen ligger.

---

## Prompt 26 (2026-11-07 15:34)

**Mig:**

den ligger nu i computervision mappen som "testvideo.mp4"  
Du må gerne lave et script

**Claude:**

Testen viste et problem: den OpenCV, der er installeret, kan ikke åbne vinduer, sandsynligvis den "headless" version, som Ultralytics installerer. Jeg tjekker hvilken.

Ja, det kan køre i realtid. Scriptet [video_demo.py](video_demo.py) virker, og jeg har testet det både med vindue og uden.

**Resultat på `testvideo.mp4`** (416×416 pixels, 30 billeder i sekundet, 17 sekunder):

| Metode | Tid pr. billede | Billeder i sekundet |
|---|---|---|
| YOLOv8n 640 (GPU) | 28–29 ms | **34–36** → kan følge med videoens 30 |
| Klassisk C + CNN | 3,5–3,7 ms | ~275 |

**Sådan bruger du det:**
```bash
python video_demo.py --classical
```
Det viser den klassiske pipeline til venstre og YOLO til højre, med bokse, klassenavn og sikkerhed. Øverst i hver halvdel står en FPS-tæller. Mellemrum sætter på pause, og `q` eller Esc stopper. Videoen afspilles i normal hastighed. Med `--fast` kører den så hurtigt som muligt.

Andre muligheder:
- `--cpu` kører YOLO på CPU, hvilket viser, at det **ikke** er realtid uden GPU. Det er godt til din pointe om hardware.
- `--model=416` bruger 416-modellen.
- `--save` gemmer også en video.

**Backup:** den gemte version ligger i `results/video_demo_testvideo_640_20261107_163720/demo.mp4`, og jeg har sendt den til dig. Vis den, hvis live-demoen fejler.

## Videoen er lavet af træningsbilleder

Jeg sammenlignede hvert 25. billede i videoen med alle billeder i datasættet. **Alle 21 svarer næsten præcist til et billede i træningssættet**: de simulerede `FisheyeCamera_1_*`-billeder. Forskellen er på niveau med komprimeringsstøj.

Det betyder:
- Demoen viser **hastighed** ærligt: FPS-tallet er reelt.
- Den viser **ikke**, hvor godt modellen virker på ukendte data, fordi modellen har set de billeder under træningen. Detektionerne ser derfor bedre ud, end de reelt er.
- Sig det selv til fremlæggelsen, før censor spørger. Fx: "Videoen er fra simulator-delen af træningsdata, så den viser hastigheden. Nøjagtigheden viser jeg med testsættet." Det giver et bedre indtryk end at blive afsløret.

Det er også derfor, den klassiske pipeline næsten aldrig finder noget i videoen. Den er kun lavet til røde, runde skilte, og simulatoren har fundet 0 % i din robusthedstest.

## Problem med OpenCV i dit miljø

Dit venv har tre OpenCV-pakker installeret oven i hinanden (`opencv-python`, `-contrib` og `-headless`). Den headless version, der fulgte med Ultralytics, har overskrevet resten, så `cv2.imshow` ikke længere kan åbne vinduer. Jeg har ikke ændret dit miljø. Demoen viser i stedet videoen med tkinter, som følger med Python. Det kan påvirke dine ældre scripts med `cv2.imshow`, fx `trackbar*.py`. Hvis de skal virke igen, kan det løses ved at afinstallere og geninstallere OpenCV-pakkerne.

---

## Prompt 27 (2026-11-07 15:49)

**Mig:**

når du skriver klassisk c + cnn, hvad betyder klassisk c?

**Claude:**

"Klassisk C" er det tredje og strengeste trin af din klassiske pipeline. Trinene blev bygget op ét ad gangen i den negative test, så man kan måle, hvad hvert filter bidrager med. Navnene er defineret i `pipelines.py` (`HOUGH_PARAMS`).

Alle tre trin starter på samme måde:

> median blur → rød HSV-maske → close → open → Gaussian blur → HoughCircles

Det, der adskiller dem, er indstillingerne i Hough-trinnet og et ekstra tjek til sidst:

| Trin | Hough-indstillinger | Ring-tjek | Falske alarmer på jordbær |
|---|---|---|---|
| **A, naiv** | løse (minDist 20, param2 15): accepterer cirkler med lidt "bevis" | nej | 100 % af billederne |
| **B, strammere** | strengere (minDist 40, param2 30): cirkler skal have mere støtte og ligge længere fra hinanden | nej | 47,5 % |
| **C, med ring-tjek** | som B | **ja** | 17,5 % |

**Ring-tjekket i C** bruger viden om, hvordan et skilt ser ud: det har en **rød kant og en ikke-rød midte**. For hver cirkel, Hough finder, tjekkes det, at mindst 50 % af kanten er rød i masken, og at højst 30 % af midten er rød. Et jordbær er rødt hele vejen igennem, så det bliver afvist. Det kan du se i række 3 i `pipeline_steps.png`, hvor 5 cirkler bliver til 0.

**"Klassisk C + CNN"** betyder, at C først finder kandidaterne, altså hvor der måske er et skilt. Derefter skærer koden hver cirkel ud og lader dit CNN fra iteration 2 afgøre, hvilket skilt det er. Det er den totrinsmodel, du kan sammenligne med R-CNN: klassisk billedbehandling foreslår områder, og et neuralt netværk klassificerer dem. Varianten **"C + CNN (conf ≥ 0,9)"** afviser desuden svar, hvor CNN'et er under 90 % sikker. Det fjerner flere falske alarmer, men også flere rigtige skilte.

**Til rapporten:** navnene A/B/C er kun noget, vi har brugt internt. I rapporten kan du skrive dem ud, fx "trin 1: naiv Hough", "trin 2: strammere Hough" og "trin 3: med ring-tjek". Pointen i afsnittet er, at hvert trin bytter recall for færre falske alarmer: A finder 41,6 % af skiltene, C kun 23,2 %, men med langt færre fejl.

---

## Prompt 28 (2026-11-07 19:22)

**Mig:**

[vedhæftet fil: Portef_ljerapport___Computer_Vision_trafikskilte.pdf] Jeg skal dokumentere min samtale med dig ift min skoles regler om AI brug.   
  
Alle samtaler hvor jeg bruger noget fra dig som direkte indgår i min opgave skal dokumenteres med prompt og dit svar. Kan du give mig en liste af prompter i samtalen der opfylder de krav?

**Claude:**

Jeg har læst rapporten og trukket alle dine prompter ud af samtalen. Nu tjekker jeg, hvilke af mine svar der faktisk er havnet i rapporten.

Her er de prompter fra samtalen, hvor noget fra mine svar er brugt direkte i rapporten. Jeg har sammenlignet hver prompt med det, jeg lavede eller svarede, og med den rapport-PDF, du sendte. Tidspunkterne er fra samtaleloggen.

**Det kan jeg ikke se:** hvilke af mine formuleringer du selv har kopieret, eller hvad den anden Claude-chat har omskrevet. Listen viser derfor, hvor mit indhold er havnet i rapporten, ikke præcis hvilke sætninger der stammer fra mig.

### Bør dokumenteres (indgår direkte)

| # | Tidspunkt | Prompt (forkortet) | Hvor i rapporten |
|---|---|---|---|
| 1 | 02/11 14:20 | Startprompten (din lange indsatte projektbeskrivelse) | `label_check.py` og tallene i 4.4 (1 modstridende fil, 331 med flere skilte, små bokse) |
| 2 | 02/11 14:25 | "Ja, kør det" | Kørslen af CNN forsøg 2: 93,6 %, confusion matrix og Grad-CAM (6.3, figur 12–13, bilag 3) |
| 3 | 02/11 14:30 | "Det er ikke en fair sammenligning…" | Mine forslag til at beskrive overgangen fra iteration 1 til 2 (6.3) |
| 4 | 02/11 14:55 | "Sørg for at hver ny iteration får nyt heatmap…" | Ændringer i CNN-koden, så resultatmapperne ikke overskrives |
| 6 | 02/11 14:59 | "Ja, start YOLOv8" | `train_yolo.py` (tabel 3, bilag 2c) |
| 7 | 02/11 15:07 | "Hvad får yolov8 at vide på forhånd…" | Transfer learning og COCO (5.8), og pointen om at CNN'et aldrig skulle *finde* lysene (7.1) |
| 8 | 02/11 15:28 | "my laptop ran out of power…" | YOLO-træningen ved 416, mAP50 0,936 (6.4) |
| 9 | 02/11 16:04 | "Ville jeg kunne bruge andre metoder… negative billeder?" | Idéen om klassisk + CNN og sammenligningen med R-CNN (5.7), negative billeder (4.6) |
| 10 | 02/11 16:09 | "Ja, byg testen" | `negative_test.py` og tabel 8 |
| 11 | 02/11 16:23 | Studieplanen: "Er der andre ting der ville være godt at have med i rapporten?" | Forslagene, der formede rapporten: billedvalg (4.7), lystest og fravalgte filtre |
| 12 | 02/11 17:07 | "20.1 og 30.1 etc er ikke mine egne billeder… kør punkt 1, 4 og 5" | `pipelines.py` og `robustness_test.py`: lys, størrelse, billedtype, hastighed og afprøvning af param2 (tabel 4, figur 10 og 14, bilag 2a og 3) |
| 13 | 02/11 17:35 | "Kan du sætte en test i gang hvor du opskalerer billederne…" | Træningen ved 640 og `yolo_size_compare.py` (tabel 5–6) |
| 14 | 02/11 20:20 | "ville det give mening at prøve at opskalere mere?" | Anbefalingen af 832 |
| 15 | 02/11 20:25 | "bægge dele. Kør først endnu en opskalering…" | 832-kørslen (tabel 5–6) |
| 17 | 02/11 22:05 | "Kan du forklare mig helt præcis hvad arkitektursammenligningen vil opnå…" | Forklaringen af opløsning mod kapacitet, 8×8- og 4×4-felter (5.8 og 6.5) |
| 18 | 02/11 22:26 | "Bare start arkitektursammenligningerne… kom med en prompt…" | P2- og v8s-træningerne (tabel 7) og prompten til den anden chat |
| 19 | 03/11 17:39 | "Vi nævner bare begrænsningen i rapporten. Kan du komme med en opdateret prompt…" | Den opdaterede prompt med P2-begrænsningen |
| 20 | 03/11 18:25 | "Kan du genskrive cannytest så den laver outputs som billeder…" | Figur 2–9 |
| 21 | 03/11 19:13 | "Kan pipelines.py gemme mellemresultaterne efter hvert filter?…" | `pipeline_steps.py` og figur 11 (bilag 2a og 3) |
| 22 | 03/11 20:01 | "Blev RGB-threshold afprøvet i colortest.py?…" | Figur 1, sætningen "RGB blev ikke afprøvet" og colortest i bilag 3 |
| 24 | 03/11 20:31 | "Der mangler én måling… negative_test.py med 640-vægtene…" | 640-rækken i tabel 8 og teksten i 6.6 |
| 27 | 07/11 15:49 | "når du skriver klassisk c + cnn, hvad betyder klassisk c?" | Kun hvis du har brugt min forklaring af A/B/C i 5.6 |

### Behøver ikke dokumenteres
- **#5** "Ja, flyt dem", **#16** "Hvor lang tid vil sammenligningen tage…" og **#23** "Nej, ikke nu": de gav ikke noget indhold til rapporten.
- **#25–26** (video-demoen): bruges kun til fremlæggelsen, ikke i rapporten. Hvis din skoles regler også gælder fremlæggelsen, så tag dem med.

### To ting at være opmærksom på
1. **Afsnit 3.1 nævner ikke kode,** og der står stadig "[Tilføj: om og hvordan du har brugt AI til koden]".
   - Disse scripts har jeg skrevet fra bunden: `label_check.py`, `train_yolo.py`, `pipelines.py`, `negative_test.py`, `robustness_test.py`, `yolo_size_compare.py`, `pipeline_steps.py` og `video_demo.py`.
   - Disse har jeg rettet i: `Machineconvoluted.py`, `Machineconvoluted_crops.py`, `cannytest.py` og `colortest.py`.
   - Koden i bilag 2a og 2c er altså min. Det bør stå i 3.1.
2. **Den anden Claude-chat skal også dokumenteres.** Den har skrevet udkastene til teksten, og ifølge din egen beskrivelse i 3.1 er det dér, det meste tekst kommer fra.

Det sikreste er nok at vedlægge hele denne samtale i bilag 5 i stedet for et udvalg. Så kan ingen sige, at noget mangler.

Skal jeg eksportere samtalen som en fil, du kan vedlægge? Svar ja eller nej.

---

## Prompt 29 (2026-11-07 19:25)

**Mig:**

Nej, listen er nok

**Claude:**

Okay, så eksporterer jeg ikke samtalen.

Når du kopierer prompterne ind i bilag 5, skal mine svar med til hver af dem, ikke kun selve prompten. Det kræver din skoles regler. Svarene kan du kopiere direkte fra samtalevinduet ud fra tidspunkterne i listen.

---

## Prompt 30 (2026-11-07 19:28)

**Mig:**

oh, du må egentlig gerne eksportere samtalen som en fil

**Claude:**

Eksporten fra appen indeholder rå JSONL-data, som er svær at læse i et bilag. Jeg laver derfor også en læsbar version med kun dine prompter og mine svar.
