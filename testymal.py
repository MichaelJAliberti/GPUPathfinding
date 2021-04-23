import yaml

# Open first file
        with open(args.exp_file1) as f:
            data = yaml.safe_load(f)
        # Establish arrays to read reformatted data into
        starts = []
        goals = []
        obstacles = []
        # Read in data
        num_agents = len(data['agents'])
        for agent in data['agents']:
            starts.append(agent['start'])
            goals.append(agent['goal'])
        size = max(data['map']['dimensions'])
        for obstacle in data['map']['obstacles']:
            obstacles.append(obstacle)
