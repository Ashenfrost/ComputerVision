# Prompt til rapportskrivning

Kopiér alt under stregen ind i en ny Claude-chat. Vedhæft gerne:
- studieordningen (PDF)
- `PorteføljeIntro.md` fra https://github.com/AllanMisasa/computer-vision-2026
- de plots du vil bruge (confusion matrices, Grad-CAM, `lys.png`, `stoerrelse_*.png`, `hough_sweep.png` osv.)
- `summary.txt`-filerne fra `results/`

Tallene nedenfor er fra kørslerne 2.–3. november 2026.

---

Du skal hjælpe mig med at skrive min porteføljerapport i faget Computer Vision (IT-teknolog, UCL, 3. semester). Rapporten er hovedproduktet; koden er bilag. Skriv på dansk.

## Sådan vil jeg arbejde med dig
- Vi skriver rapporten afsnit for afsnit. Start med at foreslå en disposition ud fra vurderingskriterierne nedenfor, og vent på at jeg godkender den.
- Skriv udkast i et sagligt, teknisk sprog som en studerende på mit niveau – ikke som en forskningsartikel og ikke med fyldord. Hellere kort og præcist.
- Opfind aldrig tal, resultater eller kilder. Brug kun tallene i denne prompt eller i filer jeg vedhæfter. Mangler et tal, så skriv [MANGLER: ...] i teksten.
- Vær ærlig om svagheder og begrænsninger i mine forsøg – de skal med i rapporten, ikke skjules.
- Ros mig ikke unødigt. Sig det hvis et afsnit er svagt, uklart eller mangler refleksion.
- Stil mig af og til et spørgsmål, der får mig til at reflektere selv (fx "hvorfor tror du ...?"), så rapporten bliver min egen forståelse og ikke kun din. Men hold ikke svar tilbage for at gøre det.
- Jeg er begynder i Python. Forklar tekniske begreber kort, når de dukker op første gang.

## Vurderingskriterier (fra underviserens PorteføljeIntro.md)
1. Valg af billeder/data – begrundet.
2. Filtre anvendt i rækkefølge – med refleksion over hvorfor, og over filtre jeg IKKE valgte.
3. Test på både positive OG negative billeder.
4. Sammenligning af flere algoritmer.
Præcision vægtes ikke højt; forståelse og refleksion vægtes. Eksamen: portefølje + 10 min præsentation + 10 min spørgsmål (80/20).

## Use case
Detektion og genkendelse af trafikskilte og trafiklys fra et kamera i en bil.

## Data
- Kaggle/Roboflow-datasæt i YOLO-format, 15 klasser: Green Light, Red Light, Speed Limit 10–120 (11 klasser), Stop. Billeder 416×416.
- Datasættet er blandet af flere kilder (fundet ud fra filnavne): GTSRB-udsnit, rigtige vejfotos, simulator-billeder (fisheye) og "road*"-billeder. I train: GTSRB 1352, vejfoto 1174, simulator 508, road 496.
- Label-tjek (mit eget script, IoU-baseret): kun 1 modstridende dobbelt-label; 331 filer har reelt flere skilte. 637 simulator-bokse er under 8 px; ca. 40–49 % af Green/Red Light-boksene er under 8 px.

## Metode 1 – Klassisk pipeline (OpenCV)
Median blur → rød HSV-maske (to intervaller: H 0–8 og 172–180, S≥100, V≥70) → morfologi (close, open, 5×5 kerne) → Gaussian blur → HoughCircles (dp 1.2, param1 100, radius 6–150) → "ring-tjek" (rød kant, ikke-rødt center).
Tre trin målt: A (løse Hough-parametre), B (strengere: minDist 40, param2 30), C (B + ring-tjek).
- Hough-parametrene blev først tunet på test-sættet (fejl). Rettet: param2 valgt ved sweep på valideringssættet (val recall 24,4 % vs. test 23,2 % ved param2=30 – tæt på hinanden, så tuningen overførte).
- Klassisk kan per design kun finde runde, røde skilte – ikke trafiklys og ikke Stop (ottekant).

## Metode 2 – Eget CNN (PyTorch)
- Iteration 1: klassifikation af hele billedet → ca. 27 % accuracy. For meget støj; netværket fik ikke at vide hvor skiltet var.
- Iteration 2: klassifikation af udsnit (crops) af skiltene → 93,6 % accuracy (bedste epoch 27). Class weights i NLLLoss, StepLR, augmentation uden horisontal flip (tal bliver spejlvendt) og uden hue-ændring (rød/grøn er selve forskellen). Red/Green Light næsten perfekte. Tilbageværende forvekslinger: 60↔80, 50↔60, 100↔120. Class weights gav 19 falske "Speed Limit 10".
- Grad-CAM-heatmaps og confusion matrix for begge iterationer.
- Kombineret med klassisk: "Klassisk C + CNN" = klassisk finder kandidatregioner, CNN klassificerer dem (to-trins pipeline, svarer til idéen bag R-CNN).

## Metode 3 – YOLOv8n (Ultralytics, transfer learning fra COCO)
Samme augmentation-fravalg som CNN. seed=0 (reproducerbart – en afbrudt kørsel havde identisk mAP ved epoch 10/20/30). 50 epochs.
- Trænet ved 416: test mAP50 0,936, mAP50-95 0,783. Svaghed: Red Light recall 0,69, Green Light misser 16.

## Negativ test (40 jordbær-billeder som "hårde negativer" – røde og runde, 100 vejudsnit uden skilte, 150 positive billeder med 185 skilte, heraf 139 med rød kant)
Andel billeder med mindst én falsk alarm, og andel skilte fundet (IoU ≥ 0,5):
| Metode | Jordbær | Vejudsnit | Fundet (alle) | Fundet (røde) | Rigtig klasse | Falske bokse på positive |
|---|---|---|---|---|---|---|
| Klassisk A (naiv) | 100 % (2308 alarmer) | 15 % | 41,6 % | 55,4 % | – | 3191 |
| Klassisk B (strammere Hough) | 47,5 % | 1 % | 34,6 % | 46,0 % | – | 169 |
| Klassisk C (+ ring-tjek) | 17,5 % | 0 % | 23,2 % | 30,9 % | – | 61 |
| Klassisk C + CNN | 17,5 % | 0 % | 23,2 % | 30,9 % | 19,5 % | 61 |
| Klassisk C + CNN (conf ≥ 0,9) | 5 % | 0 % | 11,9 % | 15,8 % | 10,3 % | 19 |
| YOLOv8n (416) | 20 % | 6 % | 93,5 % | 97,1 % | 92,4 % | 11 |
| YOLOv8n (640, endelig model) | 22,5 % (9 alarmer) | 6 % | 93,0 % | 97,1 % | 93,0 % | 8 |
Kørsel: results/negative_test_20261103_213220 (samme seed og billeder; de øvrige rækker blev reproduceret præcist). 640 vs. 416 på negativer: 1 jordbærbillede mere med falsk alarm (støj ved n=40), samme på vejudsnit, 3 færre falske bokse på positive. Opskaleringen har altså ikke gjort modellen mere tilbøjelig til falske alarmer. Gevinsten på små skilte ses ikke her, fordi 150-billeders-stikprøven har få meget små skilte – den ses i størrelsesanalysen på hele testsættet.
Klassisk alene giver ingen klasse. Pointe: hvert filtertrin bytter recall for færre falske alarmer.

## Robusthed
- Lys (lysstyrke ganget med 0,3–2,0): klassisk falder til 0,5 % ved ×0,3; YOLO holder 87 %.
- Størrelse: klassisk finder 0 % af skilte under 32 px; YOLO finder 66 % under 8 px.
- Kilde: simulator 0 % klassisk vs. 70 % YOLO.
- Hastighed pr. billede: klassisk 5,7 ms; klassisk C + CNN 8,5 ms (GPU); YOLOv8n 28 ms GPU / 57,5 ms CPU.

## Opskalering (YOLOv8n trænet ved forskellige billedstørrelser)
| imgsz | mAP50 test | mAP50-95 test | mAP50 val | Træningstid | CPU pr. billede |
|---|---|---|---|---|---|
| 416 | 0,936 | 0,783 | – | 25 min | 59 ms |
| 640 | 0,957 | 0,810 | 0,971 | 41 min | 106 ms |
| 832 (batch 16) | 0,969 | 0,828 | 0,973 | 67 min | 159 ms |

Fundet med rigtig klasse efter skiltets korteste side:
| | <8 px (n=124) | 8–16 (59) | 16–32 (80) | 32–64 (88) | 64–128 (118) | ≥128 (301) | I alt |
|---|---|---|---|---|---|---|---|
| 416 | 66 % | 86 % | 95 % | 91 % | 97 % | 99 % | 90,9 % |
| 640 | 77 % | 95 % | 94 % | 92 % | 99 % | 97 % | 93,2 % |
| 832 | 82 % | 92 % | 94 % | 97 % | 100 % | 98 % | 94,8 % |

Pointer: opskalering tilføjer ingen ny information (billederne er 416), men små skilte dækker flere felter i YOLO's stride-8-grid. Aftagende udbytte (<8 px: +11 point, så +5). Små grupper giver støj (1 skilt = 1,7 point i 8–16). 832 vs. 640 er næsten ens på valideringssættet, så en del af test-forskellen kan være tilfældig – og batch blev ændret til 16 pga. 6 GB VRAM. At teste 416-modellen ved 640 uden at træne om gav højere mAP50 (0,949) men lavere mAP50-95 (0,755): trænings- og testforhold skal matche. Kompromis: 640, fordi CPU-tiden ved 832 er under 7 billeder/s.
Vigtig korrektion: en senere reference (v8n ved 640 med batch 16) gav test mAP50 0,966 / mAP50-95 0,814 – næsten det samme som 832 (0,969 / 0,828). Det meste af forskellen mellem 640 (batch 32) og 832 (batch 16) skyldes altså batch-størrelse og tilfældighed, ikke billedstørrelsen. 416 → 640 er den reelle forbedring.

## Arkitektursammenligning – opløsning eller kapacitet?
Spørgsmål: hvorfor overses de små skilte? To hypoteser:
- Opløsning: v8n leder tidligst i et grid med 8×8 px felter (stride 8). Test: YOLOv8n-P2 (ekstra detektionslag ved stride 4).
- Kapacitet: netværket er for lille. Test: YOLOv8s (samme opbygning, 11,1 mio. parametre mod 3,2 mio.).
P2 ved 416 har samme fineste grid (104×104) som v8n ved 832.

Gruppe 1, alle ved 416, batch 32:
| Model | mAP50 test | mAP50-95 test | <8 px | 8–16 | 16–32 | 32–64 | 64–128 | ≥128 | I alt | GPU | CPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v8n | 0,936 | 0,783 | 66 % | 86 % | 95 % | 91 % | 97 % | 99 % | 90,9 % | 21 ms | 45 ms |
| v8n-P2 | 0,878 | 0,745 | 48 % | 73 % | 85 % | 86 % | 99 % | 98 % | 85,3 % | 24 ms | 64 ms |
| v8s | 0,956 | 0,809 | 69 % | 97 % | 96 % | 96 % | 99 % | 98 % | 93,0 % | 21 ms | 92 ms |

Gruppe 2, alle ved 640, batch 16 (P2 og v8s kan ikke være i 6 GB VRAM med batch 32):
| Model | mAP50 test | mAP50-95 test | <8 px | 8–16 | 16–32 | 32–64 | 64–128 | ≥128 | I alt | GPU | CPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v8n | 0,966 | 0,814 | 76 % | 90 % | 91 % | 97 % | 99 % | 99 % | 93,5 % | 26 ms | 84 ms |
| v8n-P2 | 0,931 | 0,790 | 71 % | 81 % | 81 % | 93 % | 97 % | 98 % | 90,0 % | 30 ms | 126 ms |
| v8s | 0,967 | 0,814 | 80 % | 95 % | 92 % | 97 % | 100 % | 99 % | 94,7 % | 25 ms | 195 ms |

Fortolkning:
- Kapacitet (v8s ved 416) hjælper mest på 8–32 px og når samme mAP som v8n ved 640 – men CPU-tiden (92 ms) er næsten den samme som v8n ved 640 (84–106 ms). Ingen gratis frokost.
- Skilte under 8 px hjælpes næsten ikke af kapacitet (66 → 69 %), men af opskalering (66 → 76–77 %). Det peger på opløsning som flaskehalsen for de allermindste.
- Ved 640 giver v8s næsten intet ekstra over v8n (0,967 vs. 0,966) men koster 2,3× CPU-tid. Når opløsningen er på plads, er kapacitet ikke flaskehalsen.
- P2 er dårligst i begge grupper, men forsøget er IKKE fair over for P2: der findes ingen fortrænet P2, så kun 219 af 437 vægte (rygraden) blev overført fra COCO; neck og detektionshoved startede fra bunden. P2's valideringskurve steg stadig ved epoch 50 (416: val mAP50 0,928 ved epoch 40 → 0,942 ved 50; v8s lå fladt). Konklusion: P2 er undertrænet efter 50 epochs; forsøget kan ikke afvise hypotesen om opløsning via P2. Det er en begrænsning der skal nævnes, ikke et resultat om P2-arkitekturen. Jeg har bevidst valgt ikke at træne P2 længere; det skal stå som en begrænsning og et muligt næste skridt (fx 100+ epochs).
- Hastighedsmålingerne svinger mellem kørsler (samme v8n 416-model: 45 ms CPU i én kørsel, 59 ms i en anden). Sammenlign kun hastigheder inden for samme tabel.
- Endeligt valg til use casen (bil, svag hardware): v8n ved 640 – bedste forhold mellem nøjagtighed og CPU-tid.

## Hardware
Laptop med RTX 3050 (6 GB), Windows. Begrænsninger der påvirkede forsøgene: VRAM (batch 16 ved 832), termisk throttling på batteri.

## Start
Foreslå en disposition til rapporten, der dækker alle fire vurderingskriterier, og angiv for hvert afsnit hvilke af mine resultater og figurer der hører til. Spørg mig om det du mangler.
