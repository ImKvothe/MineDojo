import minedojo
from minedojo.sim import InventoryItem

if __name__ == "__main__":
    env = minedojo.make(
    task_id="combat_spider_plains_leather_armors_diamond_sword_shield", 
    image_size=(288,512))
    
    print(env.task_prompt)
    print(env.task_guidance)

    env.reset()
    done = 0
    for x in range(200): 
        action0 = env.action_space.no_op()
        action1 = env.action_space.no_op()
        action0[0] = 1
        action1[2] = 1
        if x == 100:
            action0[4] = 6
            action1[4] = 6
        actions = [action0, action1]
        obs, reward, done, info = env.step(actions)
        print("===============")
        if (x == 100):
            print(obs)
            print(actions)
            print(reward)
        if done:
            print("Finished")
            break
    env.close()

