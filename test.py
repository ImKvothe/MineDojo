import minedojo

if __name__ == "__main__":
    env = minedojo.make(task_id="harvest_milk", image_size=(288,512))
    print(env.task_prompt)
    print(env.task_guidance)

    env.reset()
    done = 0
    for _ in range(100):
        actions = env.action_space.no_op()
        print(actions)
        actions[0] = 1
        actions[4] = 24
        print(actions)
        obs, reward, done, info = env.step(actions)
        if done:
            print("Finished")
            break
    env.close()

