import pickle
import numpy as np
from os import path
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def transformCommand(command):
    if 'RIGHT' in str(command):
       return 2
    elif 'LEFT' in str(command):
        return 1
    else:
        return 0
    pass


def get_ArkanoidData(filename):
    Frames = []
    Balls = []
    Commands = []
    PlatformPos = []
    log = pickle.load((open(filename, 'rb')))
    for sceneInfo in log:
        Frames.append(sceneInfo.frame)
        Balls.append([sceneInfo.ball[0], sceneInfo.ball[1]])
        PlatformPos.append(sceneInfo.platform)
        Commands.append(transformCommand(sceneInfo.command))

    commands_ary = np.array([Commands])
    commands_ary = commands_ary.reshape((len(Commands), 1))
    frame_ary = np.array(Frames)
    frame_ary = frame_ary.reshape((len(Frames), 1))
    data = np.hstack((frame_ary, Balls, PlatformPos, commands_ary))
    return data

# calculate the possible x posotion when ball is at y=400
def calculate(prev_ball_x, prev_ball_y, cur_ball_x, cur_ball_y):
    # change pivot to center
    prev_ball_x += 2
    prev_ball_y += 2
    cur_ball_x += 2
    cur_ball_y += 2

    if prev_ball_y > cur_ball_y:
        # use > for the coordination is up side down
        return cur_ball_x
    else:
        try:
            m = (cur_ball_y - prev_ball_y)/(cur_ball_x - prev_ball_x)
        except ZeroDivisionError:
            m = (cur_ball_y - prev_ball_y)/(cur_ball_x - prev_ball_x + 1)
        # (y - y0) = m(x - x0)
        # (x - x0) = (y - y0)/m
        # x        = (y - y0)/m + x0
        candidate = (400 - cur_ball_y)/(m if m != 0 else 1) + cur_ball_x -2
        #print("Raw estimate: ",candidate,"(x=",cur_ball_x,"m=",m,".)", end = " ")
        if candidate >= 0 and candidate <= 200:
            return candidate
        elif candidate > 200:
            if int((candidate/200)) % 2 == 1:
                return 200 - candidate % 200
            else:
                return candidate % 200
        else:
            candidate = abs(candidate)
            if int((candidate/200)) % 2 == 0:
                return candidate % 200
            else:
                return 200 - candidate % 200

if __name__ == '__main__':
    megadata = []
    for i in range(1,19):
        filename = path.join(path.dirname(__file__), 'log', '%d.pickle'%i)
        data = get_ArkanoidData(filename)
        data=data[1::]
        Balls = data[:, 1:3]
        Balls_next = np.array(Balls[1:])
        vectors = Balls_next - Balls[:-1]

        """
        estimate = []
        for i in range(len(data)-1):
            estimate.append(calculate(data[i][1], data[i][2], data[i+1][1], data[i+1][2]))
        estimate = np.array(estimate)
        estimate = estimate.reshape((len(estimate),1))
        data = np.hstack((data[1:,:],estimate))
        
        direction=[]

        for i in range(len(data)-1):
            if(vectors[i,0]>0 and vectors[i,1]>0):
                direction.append(0) #向右上為0
            elif(vectors[i,0]>0 and vectors[i,1]<0):
                direction.append(1) #向右下為1
            elif(vectors[i,0]<0 and vectors[i,1]>0):
                direction.append(2) #向左上為2
            elif(vectors[i,0]<0 and vectors[i,1]<0):
                direction.append(3) #向左下為3

        direction = np.array(direction)
        direction = direction.reshape((len(direction),1))
        data = np.hstack((data[1:,:], direction))
        """

        data = np.hstack((data[1:,:],vectors))
        if (i==1) : 
            megadata.append(data[0])
        megadata = np.vstack((megadata, data))
        

    mask = [1, 2, 3, 6, 7]

    X = megadata[:, mask] # frame, ballx, bally, platformPosX, platfromPosY, commands, (estimate,) vector_x, vector_y
    Y = megadata[:, -3]
    Ball_x = megadata[:,1]
    Ball_y = megadata[:,2]
    #Direct = data[:,-1]

    x_train , x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    platform_predict_clf = KNeighborsClassifier(n_neighbors=6).fit(x_train,y_train)        
    y_predict = platform_predict_clf.predict(x_test)
    print(y_predict)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy(正確率) ={:8.3f}%".format(accuracy*100))
    
    """
    ax = plt.subplot(111, projection='3d')  
    ax.scatter(X[Y==0][:,0], X[Y==0][:,1], X[Y==0][:,3], c='#FF0000', alpha = 1)  
    ax.scatter(X[Y==1][:,0], X[Y==1][:,1], X[Y==1][:,3], c='#2828FF', alpha = 1)
    ax.scatter(X[Y==2][:,0], X[Y==2][:,1], X[Y==2][:,3], c='#007500', alpha = 1)
    
    plt.title("KMeans Prediction")    
    ax.set_xlabel('Ball_x')
    ax.set_ylabel('Ball_y')
    ax.set_zlabel('Direction')
        
    plt.show()
    """
    
    with open(path.join(path.dirname(__file__), 'clf_KNN_BallAndDirection.pickle'), 'wb') as f:
        pickle.dump(platform_predict_clf, f)
