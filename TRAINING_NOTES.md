# Podešavanja treninga – objašnjenje

| Parametar | Vrednost | Značenje |
|-----------|----------|----------|
| **learningRate** | 0.0001 | Početna brzina učenja. Mali korak = stabilniji trening. |
| **optimizationAlgorithm** | Adam | Adaptivni optimizator – sam prilagođava LR svakoj težini. |
| **maxEpochs** | 25 | Maksimalan broj prolazaka kroz ceo dataset. |
| **earlyStopping** | true | Automatski zaustavlja trening ako nema napretka. |
| **earlyStopping.patience** | 8 | Dozvoljava 8 uzastopnih epoha bez napretka pre zaustavljanja. |
| **earlyStopping.minDelta** | 0.001 | Minimalna promena loss-a koja se računa kao napredak. |
| **earlyStopping.checkpointEpochs** | 5 | Snima model na svakih 5 epoha. |
| **maxError** | 0.01 | Stop ako greška padne ispod 1%. |
| **trainValTestSplit** | 60,10,30 | 60% trening, 10% validacija, 30% test. |
| **batchMode** | false | Bez batch-ova – sve slike se obrađuju odjednom. |
| **randomSeed** | 123 | Fiksni seed za reproduktivnost rezultata. |
| **evaluationMetrics** | Classification | Metrika za evaluaciju – tačnost, precision, recall. |
| **networkArchitecture** | networkArch1.json | Fajl sa definicijom CNN arhitekture. |
| **trainingSet** | NewImageDataset.properties | Fajl sa podešavanjima dataseta. |
| **lossFunction** | Categorical Cross-Entropy | Loss za multi-class klasifikaciju (4 klase). |
