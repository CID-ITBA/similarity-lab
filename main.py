import similab as sm
m1 = sm.load_model(model="dw2v", corpus="nyt")
lab1 = sm.Laboratory(m1)
# # lab1.findSimilars2Word("bush",1990,maxWords=10)
# ref_word= "burger"
# ref_year= "1990"
# maxWords = 10
# track_list = ["burger","pizza"]
# file = "evo_war.pdf"
# lab1.plotEvo(ref_word, ref_year, maxWords,track_list,figsize=(13,8),file=file)

lab1.sim_tests(m1.tests[0], n_neighbors=[1,3,5,10])
# print(m1.tests[1].head())
# print(type(m1.word_index))