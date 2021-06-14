def isPalindrome(x: int) -> bool:

    list1 = list(str(x))
    n = len(str(x))
    if x == 0:
        return True
    for i in range(len(list1)):
        while n > i:
            if list1[i] == list1[n-1]:
                i += 1
                n -= 1
                if i == n-1 or i == n-2:
                    return True
                else:
                    return False
            else:
                return False


print(isPalindrome(11))
