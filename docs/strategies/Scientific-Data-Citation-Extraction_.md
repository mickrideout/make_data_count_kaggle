

# **A Strategic Blueprint for Identifying and Classifying Data Citations in Scientific Literature**

## **Section 1: Deconstructing the Data Citation Challenge**

The "Make Data Count \- Finding Data References" competition presents a multifaceted Natural Language Processing (NLP) challenge that extends beyond conventional information extraction tasks. Success requires a nuanced understanding of the problem's unique structure, the heterogeneity of its targets, and the specific demands of its evaluation metric. A winning strategy must be architected around these core characteristics from the outset.

### **1.1 The Duality of the Task: Beyond Simple NER**

At its core, the competition mandates the completion of two distinct yet deeply interconnected sub-problems: the **extraction** of dataset identifiers (dataset\_id) from the full text of scientific articles, and the subsequent **classification** of each identified citation's usage context as either Primary or Secondary.1 This is not a standard Named Entity Recognition (NER) task. The two goals are fundamentally intertwined; the linguistic cues that help classify a citation's type (e.g., verbs of creation like "generated" or "collected") are often the same cues that confirm an ambiguous string is, in fact, a dataset mention and not a piece of software or a model name.

This interdependence exposes a critical vulnerability in naive pipeline architectures. A common approach might involve first running a high-precision NER model to extract all potential dataset\_id strings and then passing these extracted mentions to a separate classification model. However, this sequential process is highly susceptible to error propagation. An NER model, trained solely to identify entities, might discard a candidate with a low confidence score, thereby preventing the classifier from ever seeing it. Yet, the classifier, with access to the full sentence context, might have recognized strong contextual evidence (e.g., "the data for this study are available under accession number...") that would have validated the mention. This suggests that the most effective solutions will likely involve models that can perform these tasks jointly or in a tightly-coupled fashion, allowing information to flow between the extraction and classification components.

### **1.2 The Heterogeneity of Identifiers: A Spectrum of Structure**

A primary source of difficulty stems from the diverse nature of the target identifiers. The competition explicitly defines two broad categories: Digital Object Identifiers (DOIs) and repository-specific Accession IDs.2

* **Structured Identifiers (DOIs):** These identifiers are highly regular, typically conforming to a pattern like 10.\[prefix\]/\[suffix\], often embedded within a full URL (https://doi.org/...).2 Their consistent structure makes them highly amenable to detection using precise, rule-based methods such as regular expressions. For this subset of the problem, machine learning is not only unnecessary but potentially suboptimal, as a well-crafted regex can achieve near-perfect precision and recall.  
* **Unstructured and Semi-Structured Identifiers (Accession IDs):** This category represents the core machine learning challenge. It encompasses a vast and varied landscape of identifiers from numerous data repositories, each with its own format. Examples like "GSE12345" (from Gene Expression Omnibus), "PDB 1Y2T" (from Protein Data Bank), and "E-MEXP-568" (from ArrayExpress) illustrate this heterogeneity.2 Furthermore, the competition description notes that references to these datasets are often indirect, using variable language and appearing in diverse sections of a paper, from the methods to the references.1 This variability makes a pure rule-based approach for Accession IDs brittle, difficult to scale, and prone to failure on novel or unseen formats. It is in this domain of ambiguity and diversity that advanced, context-aware machine learning models will provide a decisive advantage.

The inherent duality of these target types dictates that a "one-size-fits-all" modeling strategy is unlikely to succeed. A single, monolithic model trained to find both perfectly-structured DOIs and wildly-variable Accession IDs would face a diluted learning objective. It would expend valuable capacity learning simple string-matching rules for DOIs while simultaneously attempting to generalize across dozens of disparate accession formats. A more effective strategy is to "divide and conquer," treating the problem as two distinct sub-problems: a solved problem of pattern matching for DOIs, and a complex learning problem for everything else. This architectural separation is a direct and necessary consequence of the data's fundamental heterogeneity.

### **1.3 The F1-Score Imperative: The Mandate for Balance**

The competition's performance is evaluated using the F1-Score, which is the harmonic mean of precision (p) and recall (r).1 The formula is given by:

F1​=2⋅p+rp⋅r​  
where precision is the ratio of true positives to all predicted positives (tp/(tp+fp)) and recall is the ratio of true positives to all actual positives (tp/(tp+fn)).3 The F1-score's use of the harmonic mean ensures that it heavily penalizes models that excel in one metric at the expense of the other. A model must achieve a balance of reasonably high precision and high recall to score well.5

This choice of metric has profound strategic implications. It contrasts sharply with similar past competitions, such as the Coleridge Initiative, which used an F0.5-score, placing a greater emphasis on precision.7 In that context, a winning strategy might involve aggressively filtering out any low-confidence predictions to avoid false positives. Here, such a strategy would be detrimental due to the F1-score's sensitivity to recall.

This is particularly relevant given the problem statement that an estimated 86% of research data remains "uncited" in formal systems, often mentioned implicitly within the text.1 To achieve a high F1-score, a model must be capable of unearthing these hidden, non-standard mentions, which is fundamentally a high-recall task. The central challenge of this competition, therefore, is not merely to build a precise classifier but to architect a system that can cast a wide net to capture potential mentions (maximizing recall) and then intelligently verify those candidates to maintain high precision. This shifts the strategic focus from simple filtering to a more sophisticated paradigm of

**high-recall candidate generation followed by high-accuracy verification**. The entire solution architecture must be designed as a funnel, where the performance of the initial, high-recall stage is as critical to the final F1-score as the performance of the final, high-precision classification model.

## **Section 2: The Foundational Layer: Preprocessing and Candidate Generation**

Before any advanced modeling can be attempted, a robust and meticulously engineered foundational layer is required to process the raw source documents and generate a comprehensive set of candidate dataset mentions. The performance ceiling of the entire system is established at this stage; a model cannot identify a dataset mention that was lost or mangled during text extraction. This section outlines a two-part strategy for maximizing data quality and initial recall.

### **2.1 Robust Text Extraction from Scientific Formats**

The competition provides articles in both PDF and XML formats, with XML being available for approximately 75% of the training and test sets.2 This dual format provides a clear path for optimizing text extraction.

* **XML-First Strategy:** For any article where an XML file is provided, it must be treated as the canonical source of text. XML offers a clean, structured representation of the document, semantically separating content from layout. It explicitly tags sections (e.g., \<abstract\>, \<methods\>, \<ref-list\>), which is invaluable for downstream feature engineering and contextual analysis.8 Parsing XML avoids the myriad pitfalls of PDF extraction, such as incorrect reading order, hyphenation artifacts, and garbled text from tables or figures.  
* **Advanced PDF Parsing:** For the \~25% of articles that are PDF-only, a sophisticated, layout-aware parsing tool is non-negotiable. Standard command-line utilities like pdftotext are inadequate for the complex, multi-column layouts common in scientific papers. A state-of-the-art parser designed for scholarly documents is essential. Promising options include:  
  * **Grobid:** A well-established machine learning library for structuring raw scientific documents.  
  * **Marker:** A more recent, transformer-based tool noted in competition discussions for its high accuracy in converting PDFs to clean Markdown, correctly handling complex layouts, headers, tables, and code snippets.9 Investment in optimizing this PDF-to-text conversion is critical, as any errors introduced here are irrecoverable.  
* **Text Cleaning and Normalization:** Once the raw text is extracted, a standardized cleaning pipeline must be applied. This includes normalizing Unicode characters (e.g., replacing smart quotes with standard ones), handling special characters, and segmenting the document into sentences. For sentence segmentation, a library optimized for scientific text, such as **ScispaCy**, is preferable to general-purpose tokenizers, as it better handles domain-specific abbreviations and structures.10 Standard preprocessing steps like stop-word removal and lowercasing should be applied judiciously.11 While lowercasing is generally beneficial, it must be done with care, as the specific casing of some Accession IDs can be a significant feature.

### **2.2 A Two-Tiered Strategy for Candidate Generation**

The objective of this stage is to generate a superset of all text spans that *could* plausibly be a dataset identifier. This process should be tuned for maximum recall, with the understanding that subsequent machine learning models will be responsible for filtering out false positives. A two-tiered approach is recommended.

* **Tier 1: High-Precision Rule-Based Extraction:** This tier uses regular expressions to capture highly structured and common identifiers with near-perfect precision. This leverages the principles of rule-based NER, which excels when entities follow predictable patterns.13  
  * **DOI Regex:** A robust regular expression to find all DOI strings, capturing variations such as those with or without the https://doi.org prefix, and different casing for the word "doi".  
  * **Common Accession ID Regex:** A curated library of regular expressions for the most frequent data repositories identified in the training data and from external sources (e.g., Gene Expression Omnibus: GSE\\d{4,}, Protein Data Bank: \[1-9\]\\w{3}, ArrayExpress: E-\\w{4,}-\\d+).  
* **Tier 2: High-Recall Heuristic-Based Search:** This tier is designed to find novel, rare, or less-structured dataset mentions that do not conform to the patterns in Tier 1\. This approach is directly inspired by successful strategies from the similar Coleridge Initiative competition, where heuristic-based candidate selection proved highly effective.15  
  * **Contextual Pattern Matching:** Using a tool like spaCy's Matcher, create rules to identify noun phrases or capitalized text spans that appear in close proximity to a set of trigger keywords. These keywords should include terms like data, dataset, database, archive, repository, accession number, ID, deposited in, available from, retrieved from, and publicly available.  
  * **Structural Location:** Target specific sections of the paper known to contain these mentions, such as "Data Availability" statements, "Methods," or figure captions.  
  * **Syntactic Filtering:** Focus on spans that have a mix of alphabetic and numeric characters, a common feature of accession codes.

This two-tiered candidate generation strategy provides an opportunity for powerful feature engineering. The method by which a candidate span is generated is itself a strong signal of its likely validity. By tagging each candidate with its origin (e.g., found\_by:DOI\_REGEX, found\_by:GEO\_REGEX, found\_by:KEYWORD\_HEURISTIC), this information can be passed to the downstream machine learning model. The model can then learn, for instance, that candidates originating from DOI\_REGEX are almost certainly true positives, whereas candidates from KEYWORD\_HEURISTIC require much stronger contextual evidence from the surrounding text to be confirmed. This transforms the candidate generation process from a simple, disconnected preprocessing step into an integrated and informative part of the feature engineering pipeline, providing the model with a strong prior to guide its learning.

## **Section 3: Core Modeling: Transformer-Based Information Extraction**

With a high-quality text corpus and a comprehensive set of candidate mentions, the next stage is to employ a sophisticated machine learning model to perform the core task of identifying and classifying dataset citations. The choice of model architecture and training methodology is paramount for achieving a competitive F1-score.

### **3.1 The Decisive Advantage of Domain-Specific Language Models**

For NLP tasks within the scientific domain, language models pre-trained specifically on scientific corpora consistently and significantly outperform their general-purpose counterparts.16 The specialized vocabulary, syntax, and semantic structures of scientific writing are not adequately captured by models trained on general web text or news articles. Therefore, the selection of a domain-specific model is a critical first step.

* **SciBERT:** A strong baseline choice, SciBERT is a BERT-based model pre-trained from scratch on a corpus of 1.14 million full-text scientific papers from computer science and biomedical fields.10 Its key advantage lies in its custom vocabulary,  
  scivocab, which is built from the scientific corpus and better represents domain-specific terms compared to BERT's basevocab. This leads to fewer out-of-vocabulary tokens and more meaningful embeddings for scientific concepts.10  
* **SciDeBERTa:** The recommended primary model backbone for this competition. SciDeBERTa builds upon the powerful DeBERTa architecture, which introduces a disentangled attention mechanism, and is then continually pre-trained on a large corpus of scientific and technical documents.16 On relevant benchmarks for scientific information extraction, such as SciERC, SciDeBERTa has demonstrated new state-of-the-art results, surpassing SciBERT.20 The disentangled attention mechanism is particularly effective for NER, as it separately models the content of a word and its relative position, allowing for a more nuanced understanding of "what" the entity is, independent of "where" it appears in the sentence.22

The clear progression in performance from general models to domain-specific models, and further to models with more advanced architectures trained on domain-specific data, provides a compelling, evidence-based rationale for selecting SciDeBERTa as the core component of the solution.

| Model | Pre-training Corpus | Key Architectural Feature | Performance on SciERC (F1) | Suitability for MDC Task |
| :---- | :---- | :---- | :---- | :---- |
| BERT-base | BooksCorpus, English Wikipedia | Transformer Encoder | N/A (General Domain) | Low: Lacks domain-specific vocabulary and context. |
| SciBERT | 1.14M Scientific Papers | Custom Scientific Vocabulary (scivocab) | 67.57 20 | High: Strong baseline due to domain-specific pre-training. |
| BioBERT | PubMed Abstracts & PMC Full-text | Biomedical Vocabulary | (Specialized for Bio-NER) | Medium: Highly effective but vocabulary is biased towards biomedical, may be less optimal for other scientific fields. |
| SciDeBERTa | General Corpus \+ Scientific Corpus | Disentangled Attention Mechanism | **72.4** 20 | **Optimal:** Combines a superior architecture with domain-specific pre-training, achieving SOTA on relevant scientific IE tasks. |

*Table 1: Comparative Analysis of Pre-trained Language Models for Scientific NER. This table synthesizes performance data to justify the selection of SciDeBERTa as the optimal backbone.*

### **3.2 Architecture: Fine-Tuning as a Sequence Tagging Task**

The most robust and common approach for this type of extraction problem is to frame it as a token-level sequence tagging task. The model is trained to predict a label for each token in an input sequence (e.g., a sentence or paragraph).

* **Labeling Schema:** The Inside-Outside-Beginning (IOB or BIO) schema is the standard for this task.23 Each token is classified as:  
  * B-DATASET: The beginning of a dataset mention.  
  * I-DATASET: Inside a dataset mention (for multi-token entities).  
  * O: Outside of any dataset mention.  
* **Model Architecture:** The recommended architecture consists of three main components built on top of the SciDeBERTa base:  
  1. **SciDeBERTa Encoder:** This base model processes the input text and generates rich, contextualized embeddings for each token.  
  2. **Linear Classification Layer:** A simple feed-forward layer is placed on top of the encoder. It takes the token embeddings as input and outputs a logit for each possible tag (B-DATASET, I-DATASET, O).  
  3. **Conditional Random Field (CRF) Layer:** This layer is added on top of the linear classifier and is a crucial component for high-performance NER.24 While the linear layer predicts tags for each token independently, the CRF layer learns the transition probabilities between tags and considers the entire sequence of labels jointly. This allows it to enforce structural constraints, such as the fact that an  
     I-DATASET tag cannot follow an O tag, or that B-DATASET must be followed by either I-DATASET or O. This ability to model label dependencies significantly reduces structurally invalid predictions and improves the overall coherence and accuracy of the extracted entities.

### **3.3 Essential Technique: Data Augmentation for Robustness**

Given the inherent noise and potential incompleteness of the training labels in a real-world dataset 9, and the vast, long-tailed distribution of possible dataset identifiers, data augmentation is not an optional enhancement but a critical technique for building a robust and generalizable model.

* **Mention Replacement:** This is arguably the most powerful augmentation technique for this task, as demonstrated by winning solutions in the similar Coleridge competition 27 and supported by NER research.28 The process involves:  
  1. Compiling a large, external dictionary of known dataset identifiers from public repositories (e.g., Dryad, Zenodo, GEO, PDB) and other scientific corpora.  
  2. During training, for a given training example, randomly replace the ground-truth dataset mention with a different identifier sampled from the external dictionary.  
     This forces the model to learn the contextual patterns that signal a dataset mention (e.g., phrases like "data are available at...") rather than simply memorizing the specific dataset names present in the limited training set.  
* **Contextual Augmentation:** To improve the model's robustness to linguistic variation, the context surrounding the entity mentions can also be augmented. Techniques like synonym replacement (e.g., changing "we collected data" to "we gathered data") or back-translation (translating the sentence to another language and back to English) can create new, semantically similar training examples.28  
* **Negative Mining:** To improve precision, it is crucial to train the model on challenging negative examples. This involves actively identifying strings within the scientific papers that resemble dataset identifiers but are not, such as software packages (e.g., "SPSS," "Stata"), model names, or chemical formulas. These "hard negatives" should be explicitly included in the training data and labeled with the 'O' tag to teach the model to distinguish them from true dataset mentions.

A sophisticated training regimen could employ these augmentation techniques in a curriculum-based manner. The model could first be trained for a few epochs on the original, un-augmented data to learn the core patterns from the ground-truth labels. Subsequently, milder forms of augmentation (like contextual augmentation) could be introduced, followed by more aggressive techniques like mention replacement in later stages of training. This graduated approach may lead to more stable convergence and a more robust final model by preventing the model from being overwhelmed by diverse augmented data at the outset.

## **Section 4: The Classification Task: Distinguishing Primary from Secondary Data Use**

Once a dataset mention has been successfully identified, the second core task is to classify its usage as either Primary (data generated for the study) or Secondary (data reused from another source).1 This is a classic binary text classification problem that hinges on detecting subtle linguistic and structural cues within the paper.

### **4.1 Framing the Task as Contextual Classification**

The classification task can be formally defined as: for each (article\_id, dataset\_id) pair identified in the extraction phase, predict a label from the set {Primary, Secondary}.

* **Input Representation:** The input to the classifier should not be the dataset identifier alone, but a window of text that provides its surrounding context. The optimal size of this window is an empirical question that requires experimentation. Plausible options include:  
  * The single sentence containing the mention.  
  * The full paragraph containing the mention.  
  * A fixed-size window of N tokens before and after the mention.  
    The paragraph-level context is likely to be the most effective, as it balances providing sufficient information without introducing excessive noise from distant text.  
* **Classifier Model:** Consistent with the strategy for NER, the classifier should be based on a transformer model pre-trained on scientific text. A fine-tuned **SciDeBERTa** model is the ideal choice, as its superior language understanding capabilities are equally beneficial for this nuanced classification task.31 The model would take the context window as input and use the embedding of the special \`\` token, passed through a linear layer with a softmax or sigmoid activation, to produce the final classification.

### **4.2 Engineering Features from Linguistic and Structural Cues**

Success in this classification task depends on the model's ability to recognize the "fingerprints" of data creation versus data reuse.33 This can be framed as a search for evidence of authorial agency versus evidence of retrieval. The model must learn to answer the implicit question: "Are the authors of this paper the

*agents* of data creation, or are they the *recipients* of existing data?"

* **Linguistic Cues for Primary Use (Creation):** These cues typically involve language that positions the authors as the creators of the data.  
  * **Keywords and Verbs of Agency:** generated, created, collected, developed, produced, sequenced, surveyed, measured.  
  * **Possessive and First-Person Phrases:** our data, our dataset, the data for this study, we report, we generated. The presence of first-person pronouns (we, our) coupled with verbs of creation is a very strong signal of primary use.35  
  * **Methodological Descriptions:** Sentences describing the process of data collection (e.g., "Participants were recruited and surveyed...") in close proximity to the dataset mention are indicative of primary data.  
* **Linguistic Cues for Secondary Use (Reuse):** These cues point to the data having an external origin, predating the current study.  
  * **Keywords and Verbs of Retrieval:** reused, reanalyzed, obtained from, downloaded from, retrieved from, acquired from, accessed.  
  * **Attributional Phrases:** data from, publicly available data from..., previously published by..., as described in \[citation\]. Citing another paper or repository as the source of the data is a definitive sign of secondary use.37  
  * **Analytical Descriptions:** Language that describes analyzing or interpreting existing data, rather than creating it, points to secondary use.39  
* **Structural Features:** The location of the mention within the document's structure is a powerful, often decisive, feature. This information is readily available from the XML files.2  
  * **Section Headers:** A mention in a "Methods" or "Data Collection" section is more likely to be primary. A mention in a "Related Work" or "Introduction" section is more likely to be secondary. A mention in a dedicated "Data Availability Statement" can be either, but the surrounding text will be highly informative. This structural information can be encoded by prepending a special token (e.g., \`\`) to the input text fed to the classifier, allowing the model to learn the strong correlation between section type and citation type.

While the competition forces a binary Primary/Secondary choice, the reality of scientific practice can be more complex. A single study might combine newly generated data with existing public data. In such cases, the text may contain cues for both classes. A well-trained classifier might reflect this ambiguity by producing a prediction with low confidence (e.g., a probability of 0.55 for Primary). This output should not be viewed merely as model uncertainty, but as a valuable signal of a potentially complex or mixed-use case. In an ensemble of models, disagreement between classifiers on a particular mention could serve a similar purpose, flagging instances that may warrant special handling or represent a fundamentally ambiguous class within the data.

## **Section 5: Advanced Architectures for Integrated Extraction**

While a pipeline of separate NER and classification models provides a strong baseline, state-of-the-art performance in complex information extraction tasks often comes from integrated architectures that solve the problem jointly. These models can mitigate the error propagation inherent in pipelines and leverage the synergistic relationship between the two subtasks.

### **5.1 The Power of Joint Entity and Relation Extraction (JERE)**

Joint Entity and Relation Extraction (JERE) models are designed to identify entities and the semantic relations between them in a single, unified process.40 This approach is a natural fit for the Make Data Count competition. The task can be reframed as extracting tuples of the form

(dataset\_id, type), which is equivalent to:

1. Identifying a DATASET entity (dataset\_id).  
2. Identifying its relationship (type) to the parent document.

By tackling these tasks simultaneously, a JERE model can learn that certain contextual words not only help classify a known entity but can also help identify an entity in the first place. For example, the phrase "we collected the..." strongly suggests that the following noun phrase is both a DATASET entity and has a Primary relationship with the paper. A joint model can leverage this dual signal, whereas a pipeline-based NER model might only see weak evidence for the entity itself.42 Various JERE frameworks have been proposed, including table-filling approaches, sequence-to-sequence models, and span-based methods.20

### **5.2 A Deep Dive into SOTA Frameworks: Multi-Head Selection and QA-based Extraction**

For this competition, two advanced architectural patterns are particularly promising due to their proven effectiveness and conceptual elegance.

* **Multi-Head Selection Architecture:** This powerful JERE framework, described by Bekoulis et al., models relation extraction as a problem where each entity can have multiple "heads," or relational links to other entities.25 The core mechanism involves using sigmoid classifiers for each potential relation, allowing for non-mutually exclusive relationships.25 A simplified but powerful adaptation for this competition would be a "tightly-coupled pipeline":  
  1. An encoder (e.g., SciDeBERTa) generates contextual token embeddings.  
  2. An NER head (e.g., a linear layer followed by a CRF) identifies all DATASET entity spans.  
  3. For each identified span, a representation is created (e.g., by pooling the embeddings of its constituent tokens).  
  4. This span representation is fed into a separate binary classification head (a linear layer with a sigmoid activation) to predict the type (Primary/Secondary).  
     This is more powerful than a fully decoupled pipeline because the classification head operates on the rich, contextual representations learned specifically for the entity extraction task, enabling a degree of shared learning.  
* **Question Answering (QA) Framing:** An elegant and increasingly popular alternative is to reframe the entire problem as extractive question answering. This approach has shown strong results in similar competitions.27 Instead of training a generic NER model, one fine-tunes a QA model (like a SciDeBERTa fine-tuned on a SQuAD-like task) to answer specific questions about the text. The pipeline would be:  
  1. For each article, pose the question: **"What datasets were generated for this study?"** The model's extracted answer spans would be labeled as Primary.  
  2. Then, pose the question: "What datasets were reused or obtained for this study?" The extracted answer spans from this query would be labeled as Secondary.  
     This approach implicitly forces the model to learn the linguistic cues for Primary and Secondary usage as part of the process of locating the answer. It is an end-to-end method that performs extraction and classification in a single step, potentially bypassing the complexities of designing explicit JERE architectures.

| Approach | Core Mechanism | Pros | Cons | Relevance to MDC Task |
| :---- | :---- | :---- | :---- | :---- |
| **Standard Pipeline** | Two independent models: (1) NER model extracts entities. (2) Classifier model predicts type for each extracted entity. | Simple to implement and debug. Can use specialized models for each subtask. | Highly susceptible to error propagation. No shared learning between tasks. | Strong baseline, but likely suboptimal for top performance due to the intertwined nature of the tasks. |
| **Tightly-Coupled Pipeline** | Shared encoder (e.g., SciDeBERTa) with two separate "heads": one for NER (token-level) and one for classification (span-level). | Reduces error propagation by sharing representations. More efficient than two full models. | Interaction between tasks is still indirect (only through shared encoder). | **High.** A pragmatic and powerful approach that balances implementation complexity with performance gains. Likely to be a very competitive strategy. |
| **Full JERE / QA Framing** | A single, end-to-end model that outputs (entity, type) tuples directly, either through a specialized JERE architecture or by answering targeted questions. | Eliminates error propagation entirely. Maximizes shared learning and leverages task synergy. QA framing is conceptually simple. | Can be more complex to implement and train. Failure modes can be harder to diagnose. | **Optimal.** Represents the state-of-the-art. The QA-based approach, in particular, offers an elegant and powerful path to a winning solution. |

*Table 2: Architectural Overview of Joint Extraction Models. This table outlines the strategic trade-offs between different levels of model integration.*

## **Section 6: Strategies for a Competitive Edge: Optimization and Ensembling**

Moving from a solid baseline to a leaderboard-topping position requires a focus on advanced techniques that maximize the chosen evaluation metric and enhance model robustness. This involves meticulous optimization, the creation of diverse model ensembles, and learning from the successes and failures of similar past competitions.

### **6.1 Maximizing the F1-Score: Strategic Thresholding**

The final F1-score is highly sensitive to the decision thresholds used to convert model probabilities into final predictions. The default threshold of 0.5 is almost never optimal. A critical final step is to tune these thresholds to maximize F1-score on a reliable local validation set.4

This process involves:

1. **Creating a High-Quality Validation Set:** This set must be held out from all training and should be representative of the test set's distribution in terms of article types and identifier diversity.  
2. **Threshold Tuning for NER:** The NER model will output confidence scores for each identified entity span. A grid search should be performed over this confidence threshold (e.g., from 0.1 to 0.99) to find the value that produces the best F1-score for entity identification on the validation set. A lower threshold will increase recall but decrease precision, and vice-versa.  
3. **Threshold Tuning for Classification:** Similarly, the classification model will output a probability for the Primary class. The threshold for this probability must also be tuned to maximize the final competition metric.

This tuning process is not a minor detail; it can lead to significant gains in the final score and is a hallmark of competitive data science.

### **6.2 The Power of the Collective: Designing a Hybrid Ensemble System**

It is a well-established principle in competitive machine learning that ensembles of diverse models are more robust and performant than any single model.44 A single model, no matter how well-trained, will have specific weaknesses and failure modes. An ensemble can mitigate these individual errors. A powerful ensemble for this task should incorporate diversity in both architecture and data.

A recommended ensemble architecture would include:

1. **Component A (Rule-Based):** A high-precision system using regular expressions to identify DOIs and a curated list of common Accession IDs.13 The outputs from this component can be treated as high-confidence ground truths that can anchor the ensemble's predictions.  
2. **Component B (Primary SOTA Model):** A fine-tuned SciDeBERTa+CRF model, as detailed in Section 3\. This will serve as the workhorse of the ensemble.  
3. **Component C (Diverse ML Model):** A second machine learning model trained to introduce diversity. This could be a SciBERT-based model, a SciDeBERTa model trained with a different random seed, or a model trained on a different subset of augmented data. Its purpose is to correct the blind spots of the primary model.  
4. **Component D (QA-based Model):** An extractive QA model, as described in Section 5\. This introduces significant architectural diversity, as it approaches the problem from a fundamentally different perspective than sequence tagging.

**Ensembling Logic:** Predictions from these diverse models can be combined using weighted voting. For instance, a candidate entity might be included in the final prediction if it is identified by the rule-based system OR by at least two of the three ML models. The classification (Primary/Secondary) can be decided by a majority vote, potentially weighted by the individual models' performance on the validation set.

A more sophisticated approach involves feature-level ensembling. Instead of just voting on the final outputs, the intermediate representations (e.g., the final hidden-layer token embeddings) from different models like SciBERT and SciDeBERTa can be concatenated. This combined, richer representation is then fed into a single final classification head (e.g., a CRF). This allows the final layer to learn from the "perspectives" of multiple models simultaneously, potentially creating a single, more powerful model than a simple voting ensemble.

### **6.3 Learning from the Past: Insights from the Coleridge Initiative Competition**

The Coleridge Initiative "Show US the Data" competition is the most relevant public precedent for this task, and its winning solutions offer invaluable strategic lessons.7

* **Domain Knowledge is Key:** The winning teams emphasized that simple models incorporating strong domain knowledge (e.g., heuristics based on dataset naming conventions) often outperformed more complex, general models.48 This validates the proposed hybrid approach that combines rule-based methods with ML.  
* **Augmentation is Crucial:** The first-place solution heavily relied on data augmentation, specifically by creating a large external list of dataset labels and replacing the majority of training labels with these new examples. This forced the model to generalize based on context rather than memorization.27  
* **Architectural Innovation:** The winning solution employed a GPT-based model framed as a sequential prediction task (predicting the start token, then using that information to predict the end token), which is a variant of the QA-style approach.27 This highlights the power of moving beyond standard NER architectures.  
* **Adaptation for F1-Score:** The Coleridge competition used an F0.5-score, rewarding precision. The winner noted filtering predictions that appeared fewer than four times in the corpus.27 For the current competition's F1-score, this aggressive precision-focused filtering must be relaxed to maintain higher recall.

The ultimate competitive advantage may lie in building a "hybrid of hybrids"—an ensemble that combines models with fundamentally different reasoning processes. A system that integrates a logical, rule-based component, a sequential, token-tagging component (NER), and a semantic, query-based component (QA) will be exceptionally robust. These models fail in different ways, and their consensus is therefore far more reliable than the output of any single approach. This diversity of methodology is a key differentiator for top-tier solutions.

## **Section 7: A Blueprint for a Top-Performing Solution**

This final section synthesizes the preceding analysis into a concrete, actionable blueprint for developing a submission capable of achieving a top rank in the competition. The strategy is built on a multi-stage, ensemble-based system designed for robustness, high recall, and high precision, all optimized for the F1-score metric.

### **7.1 The Final Mile: Post-Processing for a Flawless Submission**

The final step before generating the submission file is a critical post-processing stage. This involves rule-based cleaning and normalization to ensure the model's outputs perfectly adhere to the competition's formatting requirements and are free from obvious errors.1

* **DOI Normalization:** All extracted dataset identifiers that are DOIs must be standardized to the full https://doi.org/\[prefix\]/\[suffix\] format as required by the submission guidelines.  
* **Deduplication:** The ensemble process may generate duplicate predictions for the same dataset within a single article. The final submission file must contain only unique (article\_id, dataset\_id, type) tuples. A script must be run to remove any and all duplicates.  
* **Sanity-Checking and Heuristic Filtering:** A final layer of rules should be applied to catch clear false positives that may have slipped through the models. This includes:  
  * **Blocklist Filtering:** Remove any predicted dataset\_id that appears on a curated blocklist of common, non-dataset entities (e.g., common statistical software like "SPSS," academic phrases like "et al.", journal names).  
  * **Length-Based Filtering:** Remove predictions that are nonsensically short (e.g., one or two characters) or excessively long (e.g., more than 25 words), as these are highly likely to be errors.  
  * **Character-Based Filtering:** Remove predictions that consist solely of numbers or common words, which are unlikely to be valid dataset identifiers.

### **7.2 Proposed Multi-Stage Architecture: A Summary**

The recommended path to a winning solution is a multi-stage ensemble system that leverages the strengths of different approaches while mitigating their individual weaknesses. The overall architecture is designed as a sophisticated funnel, moving from broad candidate generation to high-confidence, verified predictions.

The table below outlines the complete, end-to-end blueprint for this system. It represents a synthesis of the strategies discussed throughout this report, providing an actionable flowchart for implementation. The core philosophy is one of **systematic risk mitigation through architectural diversity**. Every component is designed to address a specific potential failure point. Advanced PDF parsing mitigates input data corruption. Rule-based models mitigate ML failure on simple patterns. Data augmentation mitigates overfitting. The diverse ensemble (NER, QA, Rules) mitigates the risk of any single model's idiosyncratic errors. Threshold tuning mitigates a suboptimal precision-recall balance. Post-processing mitigates submission format errors. The winner of this competition will be the team that engineers the most robust and comprehensive *system*, not necessarily the team that trains the single best *model*.

| Stage | Component(s) | Goal | Key Rationale / Supporting Evidence |
| :---- | :---- | :---- | :---- |
| **1\. Text Extraction** | Advanced XML/PDF Parsers (e.g., Marker) | Convert all source documents into clean, structured text, preserving section information. | The quality of input text sets the performance ceiling. Advanced parsers are needed for the \~25% of PDF-only files to avoid data corruption. 2 |
| **2\. Candidate Generation** | Tier 1: Regex (DOIs, common IDs) Tier 2: Heuristics (Keyword-based search) | Generate a high-recall superset of all potential dataset mentions to be evaluated by ML models. | Maximizes recall early in the pipeline, ensuring potential "uncited" data mentions are not missed. 1 |
| **3\. Model A (Rule-Based)** | Curated Regex & String Matching | Achieve near-perfect precision on highly structured identifiers (DOIs, common accession numbers). | Provides a high-confidence anchor for the ensemble. Leverages the predictable structure of a subset of the targets. 2 |
| **4\. Model B (SOTA NER)** | Fine-tuned SciDeBERTa \+ CRF | Identify and classify dataset mentions using a state-of-the-art sequence tagging approach. | SciDeBERTa is the SOTA model for scientific NER. The CRF layer improves structural coherence of predictions. 16 |
| **5\. Model C (QA-Based)** | Fine-tuned SciDeBERTa (QA model) | Identify and classify mentions by asking targeted questions (e.g., "What data was generated?"). | Provides architectural diversity, forcing the model to learn context differently. A proven strategy in similar competitions. 27 |
| **6\. Ensembling & Optimization** | Weighted Voting / Stacking F1-Score Threshold Tuning | Combine predictions from diverse models to improve robustness and accuracy. Optimize decision thresholds to maximize the final F1-score. | Ensembles consistently outperform single models. Threshold tuning is critical for maximizing the specific evaluation metric. 4 |
| **7\. Post-Processing** | Normalization & Filtering Scripts | Ensure final predictions are correctly formatted, deduplicated, and free of obvious errors before submission. | A crucial final step to prevent score penalties due to formatting errors or simple, predictable false positives. 1 |

*Table 3: Blueprint for a Multi-Stage Ensemble Solution. This table provides a comprehensive, step-by-step strategic plan for building a winning system.*

#### **Works cited**

1. Make Data Count \- Finding Data References | Kaggle, accessed on June 17, 2025, [https://www.kaggle.com/competitions/make-data-count-finding-data-references](https://www.kaggle.com/competitions/make-data-count-finding-data-references)  
2. Make Data Count \- Finding Data References | Kaggle, accessed on June 17, 2025, [https://www.kaggle.com/competitions/make-data-count-finding-data-references/data](https://www.kaggle.com/competitions/make-data-count-finding-data-references/data)  
3. NLP Lab IDA | Kaggle, accessed on June 17, 2025, [https://www.kaggle.com/competitions/nlp-lab-ida](https://www.kaggle.com/competitions/nlp-lab-ida)  
4. F1-Score for Text Mining: A Deep Dive \- Number Analytics, accessed on June 17, 2025, [https://www.numberanalytics.com/blog/f1-score-for-text-mining-deep-dive](https://www.numberanalytics.com/blog/f1-score-for-text-mining-deep-dive)  
5. NLP Model Evaluation \- Metrics, Benchmarks, and Beyond \- DeconvoluteAI, accessed on June 17, 2025, [https://deconvoluteai.com/blog/evaluating-nlp-models](https://deconvoluteai.com/blog/evaluating-nlp-models)  
6. Understanding and Applying F1 Score: AI Evaluation Essentials with Hands-On Coding Example, accessed on June 17, 2025, [https://arize.com/blog-course/f1-score/](https://arize.com/blog-course/f1-score/)  
7. Coleridge Initiative \- Show US the Data \- Kaggle, accessed on June 17, 2025, [https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/)  
8. Extract Data From Research Paper Final \- Kaggle, accessed on June 17, 2025, [https://www.kaggle.com/code/parthplc/extract-data-from-research-paper-final](https://www.kaggle.com/code/parthplc/extract-data-from-research-paper-final)  
9. Make Data Count \- Finding Data References | Kaggle, accessed on June 17, 2025, [https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion)  
10. SciBERT: A Pretrained Language Model for Scientific Text \- SciSpace, accessed on June 17, 2025, [https://scispace.com/pdf/scibert-a-pretrained-language-model-for-scientific-text-1vldi6c74x.pdf](https://scispace.com/pdf/scibert-a-pretrained-language-model-for-scientific-text-1vldi6c74x.pdf)  
11. Text Preprocessing | NLP | Steps to Process Text \- Kaggle, accessed on June 17, 2025, [https://www.kaggle.com/code/abdmental01/text-preprocessing-nlp-steps-to-process-text](https://www.kaggle.com/code/abdmental01/text-preprocessing-nlp-steps-to-process-text)  
12. Text Preprocessing in NLP with Python Codes \- Analytics Vidhya, accessed on June 17, 2025, [https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/](https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/)  
13. Introduction to NER (Part I: Rule-based) \- Kaggle, accessed on June 17, 2025, [https://www.kaggle.com/code/remakia/introduction-to-ner-part-i-rule-based](https://www.kaggle.com/code/remakia/introduction-to-ner-part-i-rule-based)  
14. A rule-based named-entity recognition method for knowledge extraction of evidence-based dietary recommendations | PLOS One, accessed on June 17, 2025, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179488](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179488)  
15. Shivam-Miglani/Coleridge-Show-US-the-Data: Kaggle Competition \- GitHub, accessed on June 17, 2025, [https://github.com/Shivam-Miglani/Coleridge-Show-US-the-Data](https://github.com/Shivam-Miglani/Coleridge-Show-US-the-Data)  
16. Fine-Tuning Large Language Models for Scientific Text Classification: A Comparative Study \- arXiv, accessed on June 17, 2025, [http://arxiv.org/pdf/2412.00098](http://arxiv.org/pdf/2412.00098)  
17. Fine-Tuning Large Language Models for Scientific Text Classification: A Comparative Study, accessed on June 17, 2025, [https://arxiv.org/html/2412.00098v1](https://arxiv.org/html/2412.00098v1)  
18. allenai/scibert: A BERT model for scientific text. \- GitHub, accessed on June 17, 2025, [https://github.com/allenai/scibert](https://github.com/allenai/scibert)  
19. SciDeBERTa: Learning DeBERTa for Science Technology ..., accessed on June 17, 2025, [https://paperswithcode.com/paper/scideberta-learning-deberta-for-science](https://paperswithcode.com/paper/scideberta-learning-deberta-for-science)  
20. SciERC Benchmark (Named Entity Recognition (NER)) | Papers ..., accessed on June 17, 2025, [https://paperswithcode.com/sota/named-entity-recognition-ner-on-scierc](https://paperswithcode.com/sota/named-entity-recognition-ner-on-scierc)  
21. arXiv:2311.09860v1 \[cs.CL\] 16 Nov 2023, accessed on June 17, 2025, [https://arxiv.org/pdf/2311.09860](https://arxiv.org/pdf/2311.09860)  
22. (PDF) Transformer-Based Named Entity Recognition for Automated Server Provisioning, accessed on June 17, 2025, [https://www.researchgate.net/publication/391998332\_Transformer-Based\_Named\_Entity\_Recognition\_for\_Automated\_Server\_Provisioning](https://www.researchgate.net/publication/391998332_Transformer-Based_Named_Entity_Recognition_for_Automated_Server_Provisioning)  
23. Named Entity Recognition (NER) \- Papers With Code, accessed on June 17, 2025, [https://paperswithcode.com/task/named-entity-recognition-ner](https://paperswithcode.com/task/named-entity-recognition-ner)  
24. tothemoon10080/NER\_SciBERT: This project is a Named Entity Recognition (NER) system based on SciBERT, designed to offer an efficient and accurate NER model specifically for scientific literature. \- GitHub, accessed on June 17, 2025, [https://github.com/tothemoon10080/NER\_SciBERT](https://github.com/tothemoon10080/NER_SciBERT)  
25. Joint entity recognition and relation extraction as a multi-head ..., accessed on June 17, 2025, [https://arxiv.org/abs/1804.07847](https://arxiv.org/abs/1804.07847)  
26. S-NER: A Concise and Efficient Span-Based Model for Named Entity Recognition \- MDPI, accessed on June 17, 2025, [https://www.mdpi.com/1424-8220/22/8/2852](https://www.mdpi.com/1424-8220/22/8/2852)  
27. Coleridge Initiative \- Show US the Data | Kaggle, accessed on June 17, 2025, [https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data/discussion/248251](https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data/discussion/248251)  
28. An Experimental Study on Data Augmentation Techniques for Named Entity Recognition on Low-Resource Domains \- arXiv, accessed on June 17, 2025, [https://arxiv.org/html/2411.14551v1](https://arxiv.org/html/2411.14551v1)  
29. Are Data Augmentation Methods in Named Entity Recognition Applicable for Uncertainty Estimation? \- ACL Anthology, accessed on June 17, 2025, [https://aclanthology.org/2024.emnlp-main.1049.pdf](https://aclanthology.org/2024.emnlp-main.1049.pdf)  
30. An Analysis of Simple Data Augmentation for Named Entity Recognition \- ACL Anthology, accessed on June 17, 2025, [https://aclanthology.org/2020.coling-main.343.pdf](https://aclanthology.org/2020.coling-main.343.pdf)  
31. A Complete Process of Text Classification System Using State-of-the-Art NLP Models \- PMC, accessed on June 17, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9203176/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9203176/)  
32. 4\. Text Classification \- Practical Natural Language Processing \[Book\] \- O'Reilly Media, accessed on June 17, 2025, [https://www.oreilly.com/library/view/practical-natural-language/9781492054047/ch04.html](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/ch04.html)  
33. www.scribbr.com, accessed on June 17, 2025, [https://www.scribbr.com/working-with-sources/primary-and-secondary-sources/\#:\~:text=Primary%20research%20gives%20you%20direct,interprets%2C%20or%20synthesizes%20primary%20sources.](https://www.scribbr.com/working-with-sources/primary-and-secondary-sources/#:~:text=Primary%20research%20gives%20you%20direct,interprets%2C%20or%20synthesizes%20primary%20sources.)  
34. The reuse of public datasets in the life sciences: potential risks and rewards \- PMC, accessed on June 17, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7518187/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7518187/)  
35. Primary vs. Secondary Sources \- WGS202H5: Fundamentals of Research in Women and Gender Studies \- Research Guides at University of Toronto, accessed on June 17, 2025, [https://guides.library.utoronto.ca/WGS202utm/primary-vs-secondary-sources](https://guides.library.utoronto.ca/WGS202utm/primary-vs-secondary-sources)  
36. What Is the Difference Between a Primary and Secondary Source?, accessed on June 17, 2025, [https://www.wgu.edu/blog/what-difference-between-primary-secondary-source2304.html](https://www.wgu.edu/blog/what-difference-between-primary-secondary-source2304.html)  
37. Primary and Secondary Sources \- Scientific Research and Communication \- LibGuides at University of Connecticut, accessed on June 17, 2025, [https://guides.lib.uconn.edu/c.php?g=1067492\&p=8331550](https://guides.lib.uconn.edu/c.php?g=1067492&p=8331550)  
38. Primary vs. Secondary Sources | Difference & Examples \- Scribbr, accessed on June 17, 2025, [https://www.scribbr.com/working-with-sources/primary-and-secondary-sources/](https://www.scribbr.com/working-with-sources/primary-and-secondary-sources/)  
39. Primary vs. Secondary \- Primary Sources: A Research Guide \- Research Guides at University of Massachusetts Boston, accessed on June 17, 2025, [https://umb.libguides.com/PrimarySources/secondary](https://umb.libguides.com/PrimarySources/secondary)  
40. Joint Entity and Relation Extraction | Papers With Code, accessed on June 17, 2025, [https://paperswithcode.com/task/joint-entity-and-relation-extraction](https://paperswithcode.com/task/joint-entity-and-relation-extraction)  
41. A Relational Adaptive Neural Model for Joint Entity and Relation Extraction \- Frontiers, accessed on June 17, 2025, [https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.635492/full](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.635492/full)  
42. CARE: Co-Attention Network for Joint Entity and Relation Extraction \- arXiv, accessed on June 17, 2025, [https://arxiv.org/html/2308.12531v2](https://arxiv.org/html/2308.12531v2)  
43. ER-LAC: Span-Based Joint Entity and Relation Extraction Model with Multi-Level Lexical and Attention on Context Features \- MDPI, accessed on June 17, 2025, [https://www.mdpi.com/2076-3417/13/18/10538](https://www.mdpi.com/2076-3417/13/18/10538)  
44. EL4NER: Ensemble Learning for Named Entity Recognition via Multiple Small-Parameter Large Language Models \- arXiv, accessed on June 17, 2025, [https://arxiv.org/html/2505.23038v1](https://arxiv.org/html/2505.23038v1)  
45. Ensemble Learning for Named Entity Recognition \- SciSpace, accessed on June 17, 2025, [https://scispace.com/pdf/ensemble-learning-for-named-entity-recognition-3oms6ult9j.pdf](https://scispace.com/pdf/ensemble-learning-for-named-entity-recognition-3oms6ult9j.pdf)  
46. Ensemble Learning for Named Entity Recognition, accessed on June 17, 2025, [https://files.ifi.uzh.ch/ddis/iswc\_archive/iswc/ab/2014/raw.githubusercontent.com/lidingpku/iswc2014/master/paper/87960511-ensemble-learning-for-named-entity-recognition.pdf](https://files.ifi.uzh.ch/ddis/iswc_archive/iswc/ab/2014/raw.githubusercontent.com/lidingpku/iswc2014/master/paper/87960511-ensemble-learning-for-named-entity-recognition.pdf)  
47. Show US the Data Conference | Coleridge Initiative, accessed on June 17, 2025, [https://coleridgeinitiative.org/show-us-the-data-conference](https://coleridgeinitiative.org/show-us-the-data-conference)  
48. Show US the Data Kaggle Competition | Second Place Winner | Coleridge Initiative, accessed on June 17, 2025, [https://www.youtube.com/watch?v=tbFdvNiJolw](https://www.youtube.com/watch?v=tbFdvNiJolw)