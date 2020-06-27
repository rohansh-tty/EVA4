# Change the path

wordPath = r"C:\Users\Rohan Shetty\Desktop\S12-Assignment2\tiny-imagenet-200\tiny-imagenet-200\words.txt"

idPath = r"C:\Users\Rohan Shetty\Desktop\S12-Assignment2\tiny-imagenet-200\tiny-imagenet-200\wnids.txt"


def xtractClassID(path):
    """
    Helps in extracting class ID from wnids file
    """
    IDFile = open(path, "r")
    classes = []

    for line in IDFile:
        classes.append(line.strip())
    return classes


def xtractClassNames(path):
    """
    Helps in extracting ClassNames for that particular ID from words.txt file
    """
    wordFile = open(path, "r")
    classNames = {}
    wordID = xtractClassID(idPath)

    for line in wordFile:
        wordCls = line.strip("\n").split("\t")[0] # wordCls indicates the nXXXXXXX ID
        if wordCls in wordID: 
            classNames[wordCls] = line.strip("\n").split("\t")[1]  # Adding ClassName of a particular ID(key) as a value 
    return classNames

