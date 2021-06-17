SummVD's GitHub page

To run the experiments like in the paper we have built a user-friendly script :

Step 1 : Download via the google drive the datasets, word embedding and parameters : https://drive.google.com/drive/folders/1QjobC4w9G7nd2eva5sURUQ5Ys3s93gan?usp=sharing
Keep the same file structure, which means that you have to put every file/folder like in the google drive in a folder called SummVD.

Step 2 : Download the github SummVD.py file, and put it to the root of the SummVD folder.

Step 3 : You need to have the packages from the requirements.txt with at least the same version on the most important packages which are : gensim==3.8.3 / scipy==1.6.3 / rouge-score==0.0.4 / sklearn == 0.17.2

Step 4 : You can now run the experiments as follow :
"python SummVD.py --dataset_name "dataset_name" --size_output 0 "
With dataset_name one of the following : "duc", "reddit", "cnn", "reddit_tifu", "MN" (for Multi-News), "pubmed", "xsum"



<!--
**SummVD/SummVD** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
