import sys
import os
'''
subs = ['paneer','kaleab','juan', 'qian', 'alamin', 'aiswarya', 'professor','eddie', 'jensen', 'ding', 'sofia', 'fatme']
for s in subs:
  os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/iter0/  -bs 16 -tt handshape-model -rs 100 -mn hands-{}-100-iter0 -ts {} --save_model -rm train -ne 15 -sr 20'.format(s, s))
  os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/iter0/  -bs 16 -tt sign-model -rs 100 -mn hands-{}-100-iter0 -ts {} --save_model --freeze_embedder -rm finetune -ct left_hand -ne 5 -sr 20'.format(s, s))
  os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/iter0/  -bs 16 -tt sign-model -rs 100 -mn hands-{}-100-iter0 -ts {} --save_model --freeze_embedder -rm finetune -ct right_hand -ne 5 -sr 20'.format(s, s))
  os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/iter0/  -bs 16 -tt sign-model -rs 100 -mn hands-{}-100-iter0 -ts {} --save_model --freeze_embedder -rm finetune -ct both_hand -ne 3 -sr 20'.format(s, s))
'''
'''
#subs = ['paneer','kaleab','juan', 'qian']
#subs = ['alamin', 'aiswarya', 'professor']
gpuid = 0
iterss = 'iter2'
subs = ['eddie', 'jensen', 'ding', 'sofia', 'fatme']
for s in subs:
  #os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/{}/  -bs 16 -tt handshape-model -rs 100 -mn hands-{}-100-{} -ts {} --save_model -rm train -ne 15 -sr 20 -gpu {}'.format(iterss, s,iterss, s, gpuid))
  os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/{}/  -bs 16 -tt sign-model -rs 100 -mn hands-{}-100-{}-exactsame -ts {} --save_model --freeze_embedder -rm finetune -ct left_hand -ne 8 -sr 20 -gpu {}'.format(iterss, s,iterss, s, gpuid))
  os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/{}/  -bs 16 -tt sign-model -rs 100 -mn hands-{}-100-{}-exactsame -ts {} --save_model --freeze_embedder -rm finetune -ct right_hand -ne 8 -sr 20 -gpu {}'.format(iterss, s,iterss, s, gpuid))
  os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/{}/  -bs 16 -tt sign-model -rs 100 -mn hands-{}-100-{}-exactsame -ts {} --save_model --freeze_embedder -rm finetune -ct both_hand -ne 8 -sr 20 -gpu {}'.format(iterss,s,iterss, s, gpuid))
'''
#subs = ['alamin', 'aiswarya', 'professor']
gpuid = 1
iterss = 'iter2'
#subs = ['eddie', 'jensen', 'ding', 'sofia', 'fatme', 'professor']
subs = ['paneer','kaleab','juan', 'qian', 'alamin', 'aiswarya','eddie', 'jensen', 'ding', 'sofia', 'fatme', 'professor']
#subs = ['eddie', 'jensen', 'ding', 'sofia', 'fatme', 'professor']
#subs = ['alamin','kaleab']
for s in subs:
  os.system('python temp_run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/{}  -bs 16 -tt sign-model -rs 100 -ts {} -rm write-embedding -ne 25 -sr 20 -ct both_hand -lr 0.0001 -gpu {} -mn hands-{}-100-{} -resfile joint_learn_logit_avg'.format(iterss, s, gpuid, s, iterss))
  #os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/{}/  -bs 16 -tt handshape-model -rs 100 -mn hands-{}-100-{} -ts {} --save_model -rm train -ne 20 -sr 20 -gpu {} -lr 0.0001'.format(iterss, s,iterss, s, gpuid))
  #os.system('python run_model.py -dd /home/ahosain/workspace/asl_data_v2/asl_embedding_new/cropped_handpatches/ -hdir /home/ahosain/workspace/asl_handshape_classifier/data_merged_iters/{}/  -bs 16 -tt sign-model -rs 100 -mn hands-{}-100-{}-input-concat -ts {} --save_model --freeze_embedder -rm finetune -ct both_hand -ne 15 -sr 20 -gpu {}'.format(iterss, s,iterss, s, gpuid))
