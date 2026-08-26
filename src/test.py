if __name__ == "__main__":
    import json

    data1 = {
        "name": "Taro",
        "age": 20,
        "skills": ["Python", "C"]
    }
    data2 = {
        "name": "Hanako",
        "age": 23,
        "skills": ["communication"]
    }

    data = [data1, data2]

    with open("asd.json", "w") as f:
        json.dump(data, f, indent=4)
