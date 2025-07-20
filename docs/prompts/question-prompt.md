You are an expert in extracting and categorizing dataset mentions from research papers and policy documents. Your task is to **identify and extract all valid dataset mentions** and ensure their type is correctly classified. You to read, parse and then perform the extraction of the text in the ## **Text** section below. Provide your reponse in json format defined in the section ### **Extraction Schema**


### **What Qualifies as a Dataset?**
A dataset is a structured collection of data used for empirical research, analysis, or policy-making. Examples include:
- **Surveys & Census Data** (e.g., LSMS, DHS, national census records)
- **Indicators & Indexes** (e.g., HDI, GFSI, WDI, ND-GAIN, EPI)
- **Geospatial & Environmental Data** (e.g., OpenStreetMap, Sentinel-2 imagery)
- **Economic & Trade Data** (e.g., UN Comtrade, Balance of Payments Statistics)
- **Health & Public Safety Data** (e.g., epidemiological surveillance, crime reports)
- **Time-Series & Energy Data** (e.g., climate projections, electricity demand records)
- **Transport & Mobility Data** (e.g., road accident statistics, smart city traffic flow)
- **Other emerging dataset types** as identified in the text.
- **DOI Reference** is used to reference both datasets and papers. They are of the form  https://doi.org/[prefix]/[suffix] an example being https://doi.org/10.1371/journal.pone.0303785.

**Important:**  
If the dataset does not fit into the examples above, infer the **most appropriate category** from the context and **create a new `"data_type"` if necessary**.

### **What Should NOT Be Extracted?**
- Do **not** extract mentions that do not explicitly refer to a dataset by name, or a dataset by DOI reference.
- Do **not** extract DOI metions that refer to a paper and not a dataset. Use context to infer the DOI reference type.
- Do **not** extract mentions that do not clearly refer to a dataset, including, but not limited to:
1. **Organizations & Institutions** (e.g., WHO, IMF, UNDP, "World Bank data" unless it explicitly refers to a dataset)
2. **Reports & Policy Documents** (e.g., "Fiscal Monitor by the IMF", "IEA Energy Report"; only extract if the dataset itself is referenced)
3. **Generic Mentions of Data** (e.g., "various sources", "survey results from multiple institutions")
4. **Economic Models & Policy Frameworks** (e.g., "GDP growth projections", "macroeconomic forecasts")
5. **Legislation & Agreements** (e.g., "Paris Agreement", "General Data Protection Regulation")

### **Rules for Extraction**
1. **Classify `"type"` Correctly**
   - `"Primary"`: The dataset is used for direct analysis in the document. It is raw or processed data generated as part of this paper, specifically for this study
   - `"Secondary"`: The dataset is referenced to validate or compare findings. It is  raw or processed data derived or reused from existing records or published data sources.

   **Examples:**
   - `"The LSMS-ISA data is analyzed to assess the impact of agricultural practices on productivity."` → `"Primary"`
   -  "The data we used in this publication can be accessed from Dryad at doi:10.5061/dryad.6m3n9." → `"Primary"`
   - `"Our results align with previous studies that used LSMS-ISA."` → `"Secondary"`
   - "“The datasets presented in this study can be found in online repositories. The names of the repository/repositories and accession number(s) can be found below: https://www.ebi.ac.uk/arrayexpress/, E-MTAB-10217 and https://www.ebi.ac.uk/ena, PRJE43395.”` → `"Secondary"`.

### **Extraction Schema**
Each extracted dataset should have the following fields:
- `dataset_name`: Exact dataset name from the text (**no paraphrasing**).
- `dataset_type`: **Primary / Secondary**

### **Example Response**
- If no datasets are found, return an empty json list `[]`.
- An example response is:
[
    {
        "dataset_name": "LSMS-ISA",
        "dataset_type": "Primary"
    },
    {
        "dataset_name": "https://doi.org/10.1371/journal.pone.0303785",
        "dataset_type": "Secondary"
    }
]


## Text
{article_text}

## Response

