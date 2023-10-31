import minedojo

if __name__ == "__main__":
    env = minedojo.make(task_id="harvest_milk", image_size=(288,512))
    print(env.task_prompt)
    print(env.task_guidance)

    env.reset()
   
    for _ in range(100):
        obs, reward, done, info = env.step(env.action_space.no_op())
        if done:
            print("Finished")
            break;
    env.close()

