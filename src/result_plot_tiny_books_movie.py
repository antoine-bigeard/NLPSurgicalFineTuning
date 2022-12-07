from matplotlib import pyplot as plt

# percentages_val_only = [100,90,80,70,60,50,40,30,20,10,0]
# values_eval_only = [0.5199999809265137,0.50000000953674316,0.5059999823570251,0.503000020980835, 0.4909999966621399, 0.453000009059906, 0.4359999895095825, 0.4300000071525574, 0.4320000112056732,0.4309999942779541,0.42100000381469727]
# plt.plot(percentages_val_only,values_eval_only)
# plt.xlabel("% of amazon books dataset (= 100 - % of amazon movies) in the eval set")
# plt.ylabel("Validation accuracy of bert-tiny trained on amazon books")
# plt.show()



percentages_val_only = [100,90,80,70,60,50,40,30,20,10,0]
values_eval_only = [0.5199999809265137,0.50000000953674316,0.5059999823570251,0.503000020980835, 0.4909999966621399, 0.453000009059906, 0.4359999895095825, 0.4300000071525574, 0.4320000112056732,0.4309999942779541,0.42100000381469727]
percentages_fine_tunning = [100,90,80,70,60]
# values_eval_only = [0.5199999809265137,0.49000000953674316,0.5059999823570251,0.503000020980835, 0.4909999966621399]
values_fine_tune_all = [0.4909999966621399,0.48900002241134644,0.46000000834465027,0.4740000367164612,0.4870000183582306]
values_fine_tune_first = [0.4909999966621399,0.5060000419616699,0.5060000419616699,0.47200003266334534,0.5220000147819519]
values_fine_tune_middle = [0.4909999966621399,0.48500001430511475,0.5120000243186951,0.4780000150203705,0.5070000290870667]
values_fine_tune_last = [0.4909999966621399,0.5190000534057617,0.5040000081062317,0.5049999952316284,0.4710000157356262]
values_fine_tune_TEST = [0.5909999966621399,0.6190000534057617,0.6040000081062317,0.5049999952316284,0.4710000157356262]

plt.plot(percentages_val_only,values_eval_only, c= "royalblue", label="Eval only")
plt.plot(percentages_fine_tunning,values_fine_tune_all, c= "firebrick", label="Fine-tune all")
plt.plot(percentages_fine_tunning,values_fine_tune_first, c= "forestgreen", label="Fine-tune first")
plt.plot(percentages_fine_tunning,values_fine_tune_middle, c= "gold", label="Fine-tune middle")
plt.plot(percentages_fine_tunning,values_fine_tune_last, c= "darkorange", label="Fine-tune last")
plt.plot(percentages_fine_tunning,values_fine_tune_TEST, c= "darkorchid", label="Fine-tune pimped bert")
plt.xlabel("% of amazon books dataset (= 100 - % of amazon movies) in the eval set")
plt.ylabel("Validation accuracy of bert-tiny trained on amazon books")
plt.legend()
plt.show()