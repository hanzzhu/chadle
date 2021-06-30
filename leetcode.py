def longestPalindrome(s: str) -> str:
    result = ''
    list1 = list(s)
    left = 0
    right = 0
    for left in range(len(list1)):
        char = list1[left]
        for right in range(len(list1)):
            if right > left and char == list1[right]:
                string = ''.join(list1[left:right+1])
                # print(string)
                if isPalindrome(string) and len(string) > len(result):
                    result = string
            right += 1
        left += 1

    print(result)
    return result


def isPalindrome(s: str) -> bool:
    stringList = list(s)
    tempList = []
    while stringList:

        tempList.append(stringList.pop())

    if tempList == list(s):
        return True
    else:
        return False

# print(isPalindrome('abcba'))
longestPalindrome('abcbadifidab')
