<img src="data/drill_ship.png" style="width:300px; float:left">
<h1 style="text-align: center; color:teal">Data Science and Machine Learning for Planet Earth </h1>
<h3 style="text-align: center; color:darkgray">Assessed Coursework 2023</h3>
<h2 style="text-align: center; color:darkred">09:00 – 17:00 BST, Friday 17th of November 2023</h2>

# 📝 Task description

Your goal is to create the best possible predictor of sediment classes based on wireline log data. You are given one well. One section of the well has labels (the name of the lithologies), whereas the bottom part of the well has no labels. Your ultimate goal is to train an algorithm based on the section of the well that has labels to predict lithologies in the bottom part of the same well (where you don't have labels).

The performance of your algorithm will be determined based on how well you predict these unseen data (see more details about this in the notebook and below). A more detailed description of your task is given in the notebook.


# 🐟 Some notes about the data

Below is information about each features in your dataframe. This information could be useful to understand your task.

`DEPTH_WMSF`: This represents the depth of measurement below the seafloor, expressed in meters. It allows you to correlate geological features with depth.

`HCGR`: Total gamma ray counts, measured in counts per second (cps), indicate the overall radioactivity of the formation, often used to identify different rock types.

`HFK`: Potassium counts reflect the contribution of potassium to total gamma radiation. It's used to identify clay content and alteration in rocks.

`HTHO`: Thorium counts measure thorium's contribution to total gamma radiation, aiding in differentiating between siliciclastic and carbonate rocks.

`HURA`: Uranium counts assess uranium's contribution to total gamma radiation. It's particularly useful in organic-rich environments and for identifying hydrocarbon presence.

`IDPH`: Deep Phasor Dual Induction–Spherically Focused Resistivity measures how well electricity penetrates rocks at greater depths, indicating porosity and fluid content in ohms.

`IMPH`: Medium Phasor Dual Induction–Spherically Focused Resistivity, also in ohms, assesses electrical penetration at intermediate depths, providing insights into rock texture and fluid saturation.

`SFLU`: Shallow Phasor Dual Induction–Spherically Focused Resistivity measures electrical resistivity at shallower depths, crucial for understanding near-surface rock properties and fluid content.

`munsell_color`: This property is a `string` that contains the exact color identification based on the <a href="https://munsell.com/about-munsell-color/how-color-notation-works/how-to-read-color-chart/">Munsell color chart</a>. It describes sediment colors, aiding in identifying mineral content and environmental conditions, and thus lithology. Munsell color charts are used by sedimentologists and many other scientists. Converting the color to other system is not trivial: most of the time the color name itself is used, though conversion tables do exist (<a href="https://github.com/samdroid-apps/python-munsell">see this for instance</a>)

`lithology`: this is your target value, i.e. what you are trying to predict. It is a string representing the name of the lithology. You don't need to understand the lithology name for this assessment, but if you want to, you can easily google it to find out what each name means.


# 📊 Assessment type

The assessments are run as open-book assessments, and as such we have worked hard to create a coursework that assesses synthesis of knowledge rather than factual recall. Be aware that access to the internet,
notes or other sources of factual information in the time provided may not be too helpful and may well
limit your time to successfully synthesise the answers required. 

For your coursework, you can use any Python modules in the [standard library](https://docs.python.org/3/library/) and any modules supplied in the course `dsml4p` conda environement.

### 📁 Please do not change file names

Part of the assessment will use automatic testing, and part will be based on us reading your notebooks. Please <string style="color:red">do not</strong>: change the names of any files, give your answers in a different file than the notebook provided or the python script file (`.py`) provided, or delete some of the instructions in the notebook.

### 🤔 Plagiarism

The use of the work of another student,
past or present, constitutes plagiarism. Giving your work to another student to use may also constitute
an offence. Collusion is a form of plagiarism and will be treated in a similar manner. This is an individual
assessment and thus should be completed solely by you. The College will investigate all instances where
an assessment offence is reported or suspected, using plagiarism software, vivas and other tools, and apply
appropriate penalties to students. 

### 🔍 Disclosing the use of external sources

We provide you with a `references.md` file in your repo. Update this file with any books, websites or AI tools you use to help you in writing this code. For tutorial sites, the main URL or landing page you used will suffice. For Q&A sites (e.g. Stack Overflow/Stack Exchange), or when interacting with ChatGPT please link the specific question/response which helped you. ***Failing to disclose a used external source will be considered plagiarism***.

### 📈 Submission of your assessment

Your entire set of changes to the repository should be uploaded to GitHub before the submission deadline. For this we recommend using _one_ of the follow methods:

1.  The `git` tool on the command line. This can be achieved by running the following commands:
  ```bash
  git add .
  git commit -m "<A useful message>"
  git push origin main
  ```
Remember (on WSL) to use your PAT rather than your GitHub password.

2. A `git` based GUI such as GitHub Desktop
3. The source control tab in VS Code

We recommend you submit your work periodically during the assessment to save it online. Only the last submission will be visible and considered for marking.

# 🏫 Structure of the assessment

This assessment comprises three parts. To achieve 100%, all three parts need to be answered. **Part A** and **Part B** ***must*** be answered within the `dsml4pe-2023-assessment.ipynb` Jupyter notebook. **Part C** ***must*** be written within the `litho_estimator.py` file. Please do not rename any files within this folder. All the data you need is contained within the **data** folder.

### 🎓 Suggested Workflow

You may use any editor (e.g. Jupyter or VS Code) to write your code for the notebook. For the class code, we strongly recommend VSCode due to code completion and highlight.

Either way, you must only use the `dsml4p` environment and the standard libraries we included in this conda environment.

Keep in mind that data science is an iterative process. So, we recommend that you start with **Part A** (EDA), move on to **Part B** but if you find some interesting results, come back to **Part A** to continue your EDA and inform your decision. Only move to **Part C** once you are satisfied with the performance of your algorithm.

Always make sure that your notebook is clearly written as a record of what you have done, and justify your decisions using `markdown` cells. Also ensure that the notebook runs without any bugs as this is one of the marking criteria.

### ✅ Submission

Your entire set of changes to the repository should be uploaded to GitHub before the submission deadline. For this we recommend using _one_ of the follow methods:

1.  The `git` tool on the command line. If no additional files have been created, then this can be achieved by running the following commands:
  ```bash
  git add -u
  git commit -m "<A useful message>"
  git push -u origin main
  ```

Remember (on WSL) to use your PAT rather than your GitHub password.

  If you have created additional files to be submitted (e.g. additional code tests) then these should be staged using commands such as `git add test_answer.py` before `git commit`ing your work. 

2. A `git` based GUI such as GitHub Desktop
3. The source control tab in VS Code


# Marking scheme

This assessment is similar to one of the `Data Problems` that you are familiar with from the class. To help you, and to mark fairly, we divided it into 3 parts:

- For **Part A**, you are asked to do a proper Exploratory Data Analysis, come up with some conclusions about your data based on your observation, and justify these with graphics or numbers and some markdown text (***Total of 30 marks***).
- For **Part B**, you are asked to identify the best modelling approach for your problem and based on what you found in the data (***Total of 30 marks***).
- For **Part C**, you are asked to write a Python class that encapsulate your best algorithm (as defined in Part B). This class needs to be written in the provided file (`litho_estimator.py`) and has very clear stipulations (***Total of 30 marks***).

Things that will impact your overall marks for each section are:
<p>✅ How clean and easy to read your code is, and how well structured your notebook/your class is: this includes using markdown cells to explain your decisions if needed (don't justify all basic decisions though: the code needs to speak for itself)</p>
<p>✅ The overall performance of your algorithm at two distinct metrics:<br>
<l>
<ul>1. Obtaining as small a difference as possible between the F1 predicted on your chosen test set, and the F1 of your prediction on the unkown samples (the second file provided to you).</ul>
<ul>2. Using as few coefficients (`weights`) in your parametric model as possible, whilst maintaining a relatively high F1 score on your unseen data. This will be measured by the ratio of F1 / number of weights * 100.</ul>
</l>
.</p>
<p>✅ Whether or not you have demonstrated through code that your solution follows the best practices of data science.</p>
<br>

#### Core Marking Criteria:

- **DISTINCTION (70-100%)** The code is well written, easy to read and understand, and is well documented, with significant evidence you understand what you have written. The code shows that you have understood the conclusions from your EDA, and chosen the best approach possible to model this data as reflected in the metrics selected above. The code is well-structured, and makes appropriate use of helper functions and existing methods to avoid needlessly repeating code. At the lower end of the range, some advanced functionality may be missing, or the code may be less robust, but the code will still be generally well written and easy to understand. 

- **MERIT (60-69.99%)** The code is well written, easy to read and understand, and is well documented, with evidence you understand what you have written, but may have occasional error or misunderstandings, or may fail to implement some advanced functionality. Your modelling decisions based on your EDA are sound, but less than perfect.

- **PASS (50-59.99%)** The code is generally adequately written, but may have poor, or misguided sections. Basic functionality is mostly implemented, but more advanced points may be missing, or may miss a few key features. Your modelling decisions lead to some reasonable results, however, the score is low because the decisions are somewhat naive and not fully guided by sound EDA.

- **FAIL (0-49.99%)** The code is poorly written, difficult to read and understand, and is generally poorly documented, with little evidence that you understand what you have written. There is little connection between your EDA and the modelling decision that you have made, and you are not showing clear evidence that you understand what you are doing.
