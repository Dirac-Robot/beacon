def parse_command(command):
    tokens = []
    i = 0
    length = len(command)
    while i < length:
        # 공백 건너뛰기
        while i < length and command[i].isspace():
            i += 1
        if i >= length:
            break
        start = i
        # 변수명 또는 토큰 시작 위치 저장
        while i < length and not command[i].isspace() and command[i] != '=':
            i += 1
        if i < length and command[i] == '=':
            # 변수 할당 처리
            key = command[start:i]
            i += 1  # '=' 건너뛰기
            # 값 파싱
            if i < length:
                value, i = parse_value(command, i)
            else:
                value = ''
            tokens.append(f"{key}={value}")
        else:
            # 변수 할당이 아닌 일반 토큰 처리
            token_start = start
            while i < length and not command[i].isspace():
                i += 1
            tokens.append(command[token_start:i])
    return tokens


def parse_value(command, i):
    if i < len(command):
        if command[i] == '`':
            return parse_backtick_string(command, i)
        elif command[i] in ['[', '(', '{']:
            return parse_bracketed_value(command, i)
        else:
            # 다음 공백까지 파싱
            start = i
            while i < len(command) and not command[i].isspace():
                i += 1
            return command[start:i], i
    return '', i


def parse_backtick_string(command, i):
    assert command[i] == '`'
    start = i
    i += 1
    value = ['`']
    length = len(command)
    nesting_level = 1
    while i < length:
        c = command[i]
        if c == '\\' and i+1 < length:
            # 이스케이프된 문자 처리
            value.append(c)
            value.append(command[i+1])
            i += 2
        elif c == '`':
            value.append(c)
            i += 1
            if i < length and command[i] == '`':
                # 연속된 백틱은 단일 백틱으로 처리
                value.append(command[i])
                i += 1
            else:
                nesting_level -= 1
                if nesting_level == 0:
                    break
        elif c == '`':
            value.append(c)
            nesting_level += 1
            i += 1
        else:
            value.append(c)
            i += 1
    return ''.join(value), i


def parse_bracketed_value(command, i):
    brackets = {'[': ']', '{': '}', '(': ')'}
    opening_bracket = command[i]
    closing_bracket = brackets[opening_bracket]
    value = [opening_bracket]
    i += 1
    length = len(command)
    stack = [closing_bracket]
    while i < length and stack:
        c = command[i]
        if c == '\\' and i+1 < length:
            value.append(c)
            value.append(command[i+1])
            i += 2
        elif c == '`':
            backtick_value, i = parse_backtick_string(command, i)
            value.append(backtick_value)
        elif c in brackets:
            stack.append(brackets[c])
            value.append(c)
            i += 1
        elif c == stack[-1]:
            stack.pop()
            value.append(c)
            i += 1
        else:
            value.append(c)
            i += 1
    return ''.join(value), i
