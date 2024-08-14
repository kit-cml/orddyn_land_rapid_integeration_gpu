cd ..
make clean all
cd bin
./drug_sim -input_deck input_deck_test.txt -hill_file control/IC50_drug_control.csv -herg_file herg/drug_control.csv
# ./drug_sim -input_deck input_deck.txt -hill_file small_drugs/IC50_small_bepridil.csv -herg_file herg
# ./drug_sim -input_deck input_deck.txt -hill_file small_drugs/IC50_small_bepridil.csv -herg_file herg -cache_file ./result/init_small_bepridil.csv/state_only.csv 
# ini sepertinya bisa multiconc tapi tidak loop trough folder
# jadi kasih tau hill file langsung tapi herg kasih tau dir nya aja