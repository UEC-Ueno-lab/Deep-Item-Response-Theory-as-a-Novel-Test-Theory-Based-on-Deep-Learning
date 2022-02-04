# -*- coding: utf-8; -*-

#from janome.tokenizer import Tokenizer
import re
import csv
import pickle
import numpy as np
import chainer
from chainer import Chain, optimizers, training,Variable
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import pandas as pd
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy
from chainer.training import triggers
from chainer import serializers
from chainer import function
import random
from operator import itemgetter
from matplotlib import pyplot as plt
import math
from statistics import mean
from scipy import stats

#for real data
df = pd.read_csv(filepath_or_buffer="./data.csv", encoding="shift-jis", sep=",")
#df = pd.DataFrame(np.random.randint(0,2,size=(100, 30)))# ,columns=list('ABCD'))
#df = np.array(df_hint)
print(df)
df = df[1:200]
easy_questions = np.empty(0,dtype=np.int)
diff_questions = np.empty(0,dtype=np.int)
excellnt_students = np.empty(0,dtype=np.int)
poor_students = np.empty(0,dtype=np.int)

for i in range(len(df)):
    if len(df.iloc[i].value_counts())<3:
        continue
    incorrect = df.iloc[i].value_counts()[0]
    correct = df.iloc[i].value_counts()[1]
    correct_rate = correct / (incorrect+correct)
    if correct_rate <= 0.2:
        poor_students =np.concatenate([poor_students,np.array([i])])
    if correct_rate >= 0.8:
        excellnt_students =np.concatenate([excellnt_students,np.array([i])])



for i in range(len(df.columns)-1):
    if i==0:
        continue
    if len(df.iloc[:,i].value_counts())<2:
        continue
    incorrect = df.iloc[:,i].value_counts()[0]
    correct = df.iloc[:,i].value_counts()[1]
    correct_rate = correct / (incorrect+correct)
    if correct_rate <= 0.2:
        diff_questions =np.concatenate([diff_questions,np.array([i])])
    if correct_rate >= 0.8:
        easy_questions =np.concatenate([easy_questions,np.array([i])])

print(excellnt_students,poor_students)
print(diff_questions,easy_questions)
#for simulation data
#df = pd.read_csv(filepath_or_buffer=names  +".csv", encoding="shift-jis", sep=",")
#names = "test"
#df = pd.read_csv(filepath_or_buffer=names  +".csv", encoding="shift-jis", sep=",")
print(df)
#df = df.ix[:,[0,17, 19,  8 ,46, 44, 48, 31, 33,  5 ,34, 15, 10 ,16, 37, 28, 35,  3 ,49 , 1, 42, 25 , 6 ,50 ,23 ,45]]
#df = df.ix[:,[ 0,11,  4 ,12 ,40, 24, 41, 27, 36, 14, 47, 21, 29, 13, 26, 30, 20, 43,  9, 22, 18,  7, 38, 32,  2, 39]]
#students id
lst = [0] * len(df)
students = [[0 for i in range(len(df))] for j in range(len(df))]
for i in range(len(df)):
    students[i][i] = 1
students = np.array(students, dtype="float32")


#qustion id
items = [[0 for i in range(len(df.columns)-1)] for j in range(len(df.columns)-1)]
for i in range(len(df.columns)-1):
    items[i][i] = 1
items = np.array(items, dtype="float32")

class DeepIRT(Chain):
    def __init__(self,students_size, item_size, hidden_size, hidden_size2, hidden_size3, height,layer_size,out_size=2):
        super(DeepIRT, self).__init__(
            student_to_theta1 = L.Linear(students_size,hidden_size),
            student_to_theta2 = L.Linear(students_size,hidden_size),
            student_to_theta3 = L.Linear(students_size,hidden_size),
            student_to_theta4 = L.Linear(students_size,hidden_size),
            student_to_theta1_2 = L.Linear(hidden_size,hidden_size2),
            student_to_theta2_2 = L.Linear(hidden_size,hidden_size2),
            student_to_theta3_2 = L.Linear(hidden_size,hidden_size2),
            student_to_theta4_2 = L.Linear(hidden_size,hidden_size2),
            student_to_theta1_3 = L.Linear(hidden_size2,hidden_size3),
            student_to_theta2_3 = L.Linear(hidden_size2,hidden_size3),
            student_to_theta3_3 = L.Linear(hidden_size2,hidden_size3),
            student_to_theta4_3 = L.Linear(hidden_size2,hidden_size3),
            student_to_theta1_4 = L.Linear(hidden_size3,1),
            item_to_b1 = L.Linear(item_size, hidden_size),
            item_to_b2 = L.Linear(item_size, hidden_size),
            item_to_b3 = L.Linear(item_size, hidden_size),
            item_to_b4 = L.Linear(item_size, hidden_size),
            item_to_b1_2 = L.Linear(hidden_size, hidden_size2),
            item_to_b2_2 = L.Linear(hidden_size, hidden_size2),
            item_to_b3_2 = L.Linear(hidden_size, hidden_size2),
            item_to_b4_2 = L.Linear(hidden_size, hidden_size2),
            item_to_b1_3 = L.Linear(hidden_size2, hidden_size3),
            item_to_b2_3 = L.Linear(hidden_size2, hidden_size3),
            item_to_b3_3 = L.Linear(hidden_size2, hidden_size3),
            item_to_b4_3 = L.Linear(hidden_size2, hidden_size3),
            item_to_b1_4 = L.Linear(hidden_size3, 1),
            attention = L.Linear(item_size,out_size),
            #hidd = L.Linear(hidden_size,hidden_size),
            # classifierのLink functtion
            #hy = L.Linear(hidden_size * 4, out_size)
            hy = L.Linear(height, out_size)
        )
    def __call__(self,student,item,y,layer_size,height,model_type,ramda = 0,ramda2=0):
        lolo, th, beta = self.fwd(student,item,layer_size,height,model_type)
        xxx=0

        if model_type == "classification":
            #for i in range(len(lolo)):
            #    xxx = xxx + F.softmax_cross_entropy(lolo[i],y[i])
            xxx = F.softmax_cross_entropy(lolo,y)
            c = []
            for i in range(len(lolo)):
                if np.where(student[i].data!= 0)[0][0] in poor_students and y[i].data==1:
                    c.append(i)
                    #print( np.where(student[i].data!= 0)[0][0], np.where(item[i].data!= 0)[0][0])
                if np.where(student[i].data!= 0)[0][0] in excellnt_students and y[i].data==0:
                    c.append(i)
                    #print( np.where(student[i].data!= 0)[0][0], np.where(item[i].data!= 0)[0][0])
                if np.where(item[i].data!= 0)[0][0] in diff_questions and y[i].data==1:
                    c.append(i)
                    #print( np.where(student[i].data!= 0)[0][0], np.where(item[i].data!= 0)[0][0])
                if np.where(item[i].data!= 0)[0][0] in easy_questions and y[i].data==0:
                    c.append(i)
            xxx = xxx + 0.1*F.softmax_cross_entropy(lolo[c],y[c]) #loss function

        else:
            #xxx = F.mean_absolute_error(lolo,y)
            for i in range(len(lolo)):
                xxx = xxx + F.mean_absolute_error(lolo[i],y[i])
                if np.where(student[i].data!= 0)[0][0] in poor_students and y[i].data==1:
                    xxx = xxx + 0.1 * F.mean_absolute_error(lolo[i],y[i])
                    #print( np.where(student[i].data!= 0)[0][0], np.where(item[i].data!= 0)[0][0])
                if np.where(student[i].data!= 0)[0][0] in excellnt_students and y[i].data==0:
                    xxx = xxx + 0.1 * F.mean_absolute_error(lolo[i],y[i])
                    #print( np.where(student[i].data!= 0)[0][0], np.where(item[i].data!= 0)[0][0])
                if np.where(item[i].data!= 0)[0][0] in diff_questions and y[i].data==1:
                    xxx = xxx + 0.1 * F.mean_absolute_error(lolo[i],y[i])
                    #print( np.where(student[i].data!= 0)[0][0], np.where(item[i].data!= 0)[0][0])
                if np.where(item[i].data!= 0)[0][0] in easy_questions and y[i].data==0:
                    xxx = xxx + 0.1 * F.mean_absolute_error(lolo[i],y[i])
                    #print( np.where(student[i].data!= 0)[0][0] , np.where(item[i].data!= 0)[0][0])

            xxx = xxx / len(lolo)


        zeros = np.zeros((len(th),1),dtype = "float32")
        zeros = Variable(zeros)

        likeli = F.mean_absolute_error(((th**2 * -1) / 2)-0.893,zeros)
        mean = F.mean(th)
        mean2 = np.full((len(th),1),mean.data,dtype="float32")
        sd = F.mean_squared_error(th,mean2)
        #sd = sd.reshape(len(th),1)
        sd = sd ** (1/2) -1
        zero = Variable(np.array(0,dtype="float32"))
        sd = F.absolute_error(sd ,zero)
        #print(lolo,y,xxx)
        #loss_likelihood.append(likeli)
        #sg = stats.kstest(th.data, stats.norm(loc=0,scale = 1).cdf)[0]
        #sg = Variable(np.array(sg,dtype ="float32"))
        #sg = F.absolute_error(sg,zero)


        print("cross_entropy   ",xxx, "   likelihood   ",  likeli, "standard", sd)
        return xxx + likeli * ramda + sd * ramda2, xxx,likeli,sd, beta

        #return xxx
        #return F.softmax_cross_entropy(self.fwd(student,item),y)
        #return F.mean_squared_error(self.fwd(student,item),y)

    def likelihood(self,scala):
        like = 0
        for calu in range(len(scala)):
            #print(scala.data[calu])
            like = like - math.log((1/np.sqrt(2*math.pi) ) * math.exp(- scala.data[calu] * scala.data[calu] / 2))
        like = like / len(scala)
        return like


    def fwd(self,student, item,layer_size,height,model_type):
        #theta = F.sigmoid(self.student_to_theta(student))
        #b = F.sigmoid(self.item_to_b(item))
        if layer_size == 4:
            if height == 1:
                theta1 = F.tanh(self.student_to_theta1(student))
                theta1_2 = F.tanh(self.student_to_theta1_2(theta1))
                theta1_3 = F.tanh(self.student_to_theta1_3(theta1_2))
                theta1_4 = self.student_to_theta1_4(theta1_3)
                theta = theta1_4
                b1 = F.tanh(self.item_to_b1(item))
                b1_2 = F.tanh(self.item_to_b1_2(b1))
                b1_3 = F.tanh(self.item_to_b1_3(b1_2))
                b1_4 = self.item_to_b1_4(b1_3)
                b = b1_4


        if layer_size == 3:
            if height == 4:
                theta1 = F.relu(self.student_to_theta1(student))
                theta2 = F.relu(self.student_to_theta2(student))
                theta3 = F.relu(self.student_to_theta3(student))
                theta4 = F.relu(self.student_to_theta4(student))
                theta1_2 = F.relu(self.student_to_theta1_2(theta1))
                theta2_2 = F.relu(self.student_to_theta2_2(theta2))
                theta3_2 = F.relu(self.student_to_theta3_2(theta3))
                theta4_2 = F.relu(self.student_to_theta4_2(theta4))
                theta1_3 = F.relu(self.student_to_theta1_3(theta1_2))
                theta2_3 = F.relu(self.student_to_theta2_3(theta2_2))
                theta3_3 = F.relu(self.student_to_theta3_3(theta3_2))
                theta4_3 = F.relu(self.student_to_theta4_3(theta4_2))
                theta = F.concat((theta1_3,theta2_3,theta3_3,theta4_3), axis=1)

                b1 = F.relu(self.item_to_b1(item))
                b2 = F.relu(self.item_to_b2(item))
                b3 = F.relu(self.item_to_b3(item))
                b4 = F.relu(self.item_to_b4(item))
                b1_2 = F.relu(self.item_to_b1_2(b1))
                b2_2= F.relu(self.item_to_b2_2(b2))
                b3_2 = F.relu(self.item_to_b3_2(b3))
                b4_2 = F.relu(self.item_to_b4_2(b4))
                b1_3 = F.relu(self.item_to_b1_2(b1_2))
                b2_3 = F.relu(self.item_to_b2_2(b2_2))
                b3_3 = F.relu(self.item_to_b3_2(b3_2))
                b4_3 = F.relu(self.item_to_b4_2(b4_2))
                b = F.concat((b1_3,b2_3,b3_3,b4_3), axis=1)

            if height == 1:
                theta1 = F.tanh(self.student_to_theta1(student))
                theta1_2 = F.tanh(self.student_to_theta1_2(theta1))
                theta1_3 = self.student_to_theta1_3(theta1_2)
                theta = theta1_3
                b1 = F.tanh(self.item_to_b1(item))
                b1_2 = F.tanh(self.item_to_b1_2(b1))
                b1_3 = self.item_to_b1_3(b1_2)
                b = b1_3

        if layer_size == 2:
            if height ==4:
                theta1 = F.relu(self.student_to_theta1(student))
                theta2 = F.relu(self.student_to_theta2(student))
                theta3 = F.relu(self.student_to_theta3(student))
                theta4 = F.relu(self.student_to_theta4(student))
                theta1_2 = F.relu(self.student_to_theta1_2(theta1))
                theta2_2 = F.relu(self.student_to_theta2_2(theta2))
                theta3_2 = F.relu(self.student_to_theta3_2(theta3))
                theta4_2 = F.relu(self.student_to_theta4_2(theta4))
                theta = F.concat((theta1_2,theta2_2,theta3_2,theta4_2), axis=1)

                b1 = F.relu(self.item_to_b1(item))
                b2 = F.relu(self.item_to_b2(item))
                b3 = F.relu(self.item_to_b3(item))
                b4 = F.relu(self.item_to_b4(item))
                b1_2 = F.relu(self.item_to_b1_2(b1))
                b2_2= F.relu(self.item_to_b2_2(b2))
                b3_2 = F.relu(self.item_to_b3_2(b3))
                b4_2 = F.relu(self.item_to_b4_2(b4))
                b = F.concat((b1_2,b2_2,b3_2,b4_2), axis=1)
            if height == 1:
                theta1 = F.tanh(self.student_to_theta1(student))
                theta1_2 = self.student_to_theta1_2(theta1)
                theta = theta1_2

                b1 = F.tanh(self.item_to_b1(item))
                b1_2 = self.item_to_b1_2(b1)
                b = b1_2

        if layer_size == 1:
            theta1 = F.relu(self.student_to_theta1(student))
            theta2 = F.relu(self.student_to_theta2(student))
            theta3 = F.relu(self.student_to_theta3(student))
            theta4 = F.relu(self.student_to_theta4(student))
            theta = F.concat((theta1,theta2,theta3,theta4), axis=1)

            b1 = F.relu(self.item_to_b1(item))
            b2 = F.relu(self.item_to_b2(item))
            b3 = F.relu(self.item_to_b3(item))
            b4 = F.relu(self.item_to_b4(item))
            b = F.concat((b1,b2,b3,b4), axis=1)


        #b = F.concat((b1,b2,b3), axis=1)
        combined = theta - b

        if model_type == "classification":
            #alpha = F.softmax(self.attention(item))
            #out = self.hy(combined)*alpha
            out = self.hy(combined)
        else:
            alpha = self.attention(item)
            combined = alpha * combined
            combined = F.sigmoid(combined)
            out =  F.sigmoid((combined-0.5)*300)
            #out = combined

        return out, theta, b






n = len(students)
n2 = len(items)

layer = 3
height_num = 1
m_type = "classification"

#for １epoch
bs = int(n)
bs2 = int(n2)
print(bs,bs2)
EPOCH_NUM =300
losses =[]
losses2=[]
losses3=[]
losses4=[]
hidden1=50
hidden2=50
hidden3=1



model = DeepIRT(students_size = n,item_size= n2,hidden_size= hidden1,hidden_size2 = hidden2, hidden_size3 = hidden3,height = height_num ,layer_size = layer)
optimizer = optimizers.Adam()
optimizer.setup(model)

for j in range(EPOCH_NUM):
    if j % 2 == 0:
        model.student_to_theta1.enable_update()
        model.student_to_theta2.enable_update()
        model.student_to_theta3.enable_update()
        model.student_to_theta4.enable_update()
        model.item_to_b1.disable_update()
        model.item_to_b2.disable_update()
        model.item_to_b3.disable_update()
        model.item_to_b4.disable_update()
        #if layer > 1:
            #model.student_to_theta1_2.enable_update()
            #model.student_to_theta2_2.enable_update()
            #model.student_to_theta3_2.enable_update()
            #model.student_to_theta4_2.enable_update()
            #model.item_to_b1_2.disable_update()
            #model.item_to_b2_2.disable_update()
            #model.item_to_b3_2.disable_update()
            #model.item_to_b4_2.disable_update()
        #if layer > 2:
            #model.student_to_theta1_3.enable_update()
            #model.student_to_theta2_3.enable_update()
            #model.student_to_theta3_3.enable_update()
            #model.student_to_theta4_3.enable_update()
            #model.item_to_b1_3.disable_update()
            #model.item_to_b2_3.disable_update()
            #model.item_to_b3_3.disable_update()
            #model.item_to_b4_3.disable_update()
        print("now_theta")

        index = np.random.permutation(n)
        #index = range(n)
        #ここで，ミニバッチ処理
        for i in range(0,n,bs):
            st = []
            it = []
            answers = []

            #sss = []
            #answers = np.zeros(2*bs).reshape(bs,2).astype(np.int32)
            for al in range(bs):
                if al + i >= n:
                    continue
                student = index[al+i]
                for question in range(n2):
                    if pd.isnull(df.iloc[student,question + 1]):
                        continue
                    if df.iloc[student,question + 1]==-1:
                        continue

                    st.append(student)
                    it.append(question)
                    answers.append(df.iloc[student,question + 1])
                 #sss.append(student)
                    #print(al,question,df.iloc[al,question + 1])
                #stude, ite = divmod(index[i + al] , len(items))
                #st.append(stude)
                #it.append(ite)
                #answers.append(df.iloc[stude,ite + 1])

            student = students[st,]
            item = items[it,]


            if m_type == "classification":
                answers = np.array(answers, dtype="int32")
            else:
                answers = np.array(answers, dtype="float32")
                answers = answers.reshape([len(answers),1])

            x =Variable(student)
            x2 = Variable(item)
            y = Variable(answers)

            model.cleargrads()
            lo = model(x,x2,y,layer,height_num,m_type)
            loss = lo[0]
            loss.backward()

            optimizer.update()

        losses.append(loss.data)
        #losses2 = lo[4].data
        #losses3.append(lo[2].data)
        #losses4.append(lo[3].data)

    else:
        model.student_to_theta1.disable_update()
        model.student_to_theta2.disable_update()
        model.student_to_theta3.disable_update()
        model.student_to_theta4.disable_update()
        model.item_to_b1.enable_update()
        model.item_to_b2.enable_update()
        model.item_to_b3.enable_update()
        model.item_to_b4.enable_update()
        #if layer > 1:
        #    model.student_to_theta1_2.disable_update()
        #    model.student_to_theta2_2.disable_update()
        #    model.student_to_theta3_2.disable_update()
        #    model.student_to_theta4_2.disable_update()
        #    model.item_to_b1_2.enable_update()
        #    model.item_to_b2_2.enable_update()
        #    model.item_to_b3_2.enable_update()
        #    model.item_to_b4_2.enable_update()
        #if layer > 2:
        #    model.student_to_theta1_3.disable_update()
        #    model.student_to_theta2_3.disable_update()
        #    model.student_to_theta3_3.disable_update()
        #    model.student_to_theta4_3.disable_update()
        #    model.item_to_b1_3.enable_update()
        #    model.item_to_b2_3.enable_update()
        #    model.item_to_b3_3.enable_update()
        #    model.item_to_b4_3.enable_update()

        print("now_b")

        index = np.random.permutation(n2)
        #index = range(n2)
        #ここで，ミニバッチ処理
        for i in range(0,n2,bs2):
            st = []
            it = []
            answers = []
            #sss = []
            #answers = np.zeros(2*bs).reshape(bs,2).astype(np.int32)
            for al in range(bs2):
                if al + i >= n2:
                    continue
                question = index[al+i]
                for student in range(n):
                    if pd.isnull(df.iloc[student,question + 1]):
                        continue
                    if df.iloc[student,question + 1]==-1:
                        continue


                    #if student in studentsA and question in itemsA:
                    #    continue
                    #if student in studentsB and question in itemsB:
                    #    continue
                    #if question in na_matrix[student,]:
                        #print("forTest1")
                        #continue
                    st.append(student)
                    it.append(question)

                    answers.append(df.iloc[student,question + 1])
                #sss.append(item[0:10])
                    #print(al,question,df.iloc[al,question + 1])
                #stude, ite = divmod(index[i + al] , len(items))
                #st.append(stude)
                #it.append(ite)
                #answers.append(df.iloc[stude,ite + 1])

            student = students[st,]
            item = items[it,]


            #分類
            if m_type == "classification":
                answers = np.array(answers, dtype="int32")
            else:
                answers = np.array(answers, dtype="float32")
                answers = answers.reshape([len(answers),1])


            x =Variable(student)
            x2 = Variable(item)
            y = Variable(answers)

            model.cleargrads()
            lo = model(x,x2,y,layer,height_num,m_type)
            loss = lo[0]
            loss.backward()
            optimizer.update()

        losses.append(loss.data)
        #losses2 = lo[4].data
            #losses3.append(lo[2].data)
            #losses4.append(lo[3].data)
        #print("Student = " , model.student_to_theta1.W.data)
        #print( "Item = ",model.item_to_b2.W.data)
    print( "Loss = ",loss)
    if len(losses) > 100:
        print(np.average(losses[len(losses)-11:len(losses)-2]) - np.average(losses[len(losses)-10:len(losses)-1]))
        if np.average(losses[len(losses)-11:len(losses)-2]) - np.average(losses[len(losses)-10:len(losses)-1]) < 0.00001:
            break
    print(j)
        #del bin_list
        #gc.collect()

#showing loss plot
plt.plot(losses)
#plt.plot(losses2)
#plt.plot(losses3)
#plt.plot(losses4)
plt.show()

#creating model
#chainer.serializers.save_npz("./person100_question0_Order.npz", model)
#chainer.serializers.save_npz("./model/"+names+"_filtered.npz", model)





np.savetxt("the.csv",model.student_to_theta1.W.data,delimiter=",")
np.savetxt("b.csv",model.student_to_theta1.b.data,delimiter=",")
np.savetxt("the2.csv",model.student_to_theta1_2.W.data,delimiter =",")
np.savetxt("b2.csv",model.student_to_theta1_2.b.data,delimiter=",")
np.savetxt("the3.csv",model.student_to_theta1_3.W.data,delimiter =",")
np.savetxt("b3.csv",model.student_to_theta1_3.b.data,delimiter=",")

np.savetxt("item.csv",model.item_to_b1.W.data,delimiter=",")
np.savetxt("item_b.csv",model.item_to_b1.b.data,delimiter=",")
np.savetxt("item2.csv",model.item_to_b1_2.W.data,delimiter =",")
np.savetxt("item_b2.csv",model.item_to_b1_2.b.data,delimiter=",")
np.savetxt("item3.csv",model.item_to_b1_3.W.data,delimiter =",")
np.savetxt("item_b3.csv",model.item_to_b1_3.b.data,delimiter=",")
np.savetxt("attention.csv",model.attention.W.data,delimiter=",")
np.savetxt("attention_b.csv",model.attention.b.data,delimiter=",")

np.savetxt("hy.csv",model.hy.W.data,delimiter=",")
np.savetxt("hy_b.csv",model.hy.b.data,delimiter=",")

if n > n2:
    nnn = n2
else:
    nnn = n

#ここに明日やる
ok = 0
#st = np.array(student[0])
#it = np.array(items[na_matrix[0,0]])
st =[]
it = []
answers = []

#for i in range(n):
#    for j in range(len(na_matrix[i,])):
        #print(i,na_matrix[i,],df.iloc[i,int(na_matrix[i,j]+ 1)])
#        answers.append(df.iloc[i,int(na_matrix[i,j]+ 1)])

        #if i == 0 & j == 0:
        #    pare = pare + 1
        #    continue
        #st = np.array((st,student[i]),dtype="float32")
#        st.append(students[i])
        #print(it,items[na_matrix[i,j]])
        #it = np.array((it,items[na_matrix[i,j]]),dtype="float32")
#        it.append(items[na_matrix[i,j]])




#Nullを除く
#NullOrNot = df.isnull()

#for i in range(len(NullOrNot)):
#    for j in range(1,len(NullOrNot.columns)):
#        if NullOrNot.iloc[i,j]==True:
#            continue
#        st.append(students[i])
#        it.append(items[j-1])
#        answers.append(df.iloc[i,j])

#全てのデータの予測値

"""
for i in range(n):
    for j in range(n2):
        st.append(students[i])
        it.append(items[j-1])


st2 = np.array(st,dtype="float32")
it2 = np.array(it,dtype="float32")
xx = Variable(st2)
xt = Variable(it2)
yy, dumy = model.fwd(xx,xt,layer,height = height_num )
ans = yy.data
print("OK")

ans = yy.data
nrow, ncol = ans.shape


matrix = np.random.randn(n,n2)
cou = 0
for j in range(n2):
    for i in range(n):
        cls = np.argmax(ans[cou,:])
        matrix[i][j] = cls
        cou = cou + 1

    #print(np.where(st[i] == 1),np.where(it[i] ==1),cls,answers[i])
    #if cls == answers[i]:
        #ok = ok + 1

matrix = pd.DataFrame(matrix)
matrix.to_csv("Result.csv")
"""

    #print(ok,len(answers),ok /len(answers))

#chainer.serializers.save_npz("/Users/ryo/Library/Mobile Documents/com~apple~CloudDocs/DeepIRT/Classi//model/"+names+".npz", model)
#xx = Variable(students[0:nnn])
#xt2=Variable(items[0:nnn])
#yy =model.fwd(xx,xt2,layer)

#ans = yy.data
#nrow, ncol = ans.shape
#ok = 0
#for i in range(nrow):
#    cls = np.argmax(ans[i,:])
#    print(ans[i,:],cls,df.iloc[i,i+ 1])
#    if cls == df.iloc[i,i+1]:
#        ok = ok + 1

#print(ok / nnn)
