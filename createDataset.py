from PIL import Image
from game.game import Game 
import numpy as np
import csv
import tqdm


def saveData(path, episodes, steps_per_episodes, levels):
    assert steps_per_episodes <= 100
    for level in levels:
        game = Game(level)
        data = [0 for n in range(steps_per_episodes)]
        for j in tqdm.tqdm(range(episodes)):
            action, image = game.randomStep()
            Image.fromarray(image.astype(np.uint8)).resize((80, 80)).save("data/frames/" + path + "/lvl"+ str(level)+ "_" +str(j)+"_00.png")
            for i in range(steps_per_episodes):
                action, image = game.randomStep()
                data[i] = action
                Image.fromarray(image.astype(np.uint8)).resize((80, 80)).save("data/frames/" + path + "/lvl"+ str(level)+ "_"+ str(j) +"_{0:02}.png".format(i+1))
            with open("data/frames/" + path + "/lvl"+ str(level)+ "_"+str(j)+"_actions", mode='w') as action_file:
                for d in data:
                    action_file.write(str(d)+'\n')
            game.game.init() 

if __name__ == "__main__":
    # saveData(128,8,[x for x in range(15)])
    saveData("train",4096,8,range(5))
    saveData("validation",1,8,range(5,15))