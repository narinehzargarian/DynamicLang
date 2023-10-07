import copy
from enum import Enum
from env_v3 import EnvironmentManager, SymbolResult
from func_v3 import FunctionManager
from intbase import InterpreterBase, ErrorType
from tokenize import Tokenizer

# Enumerated type for our different language data types
class Type(Enum):
  INT = 1
  BOOL = 2
  STRING = 3
  VOID = 4
  FUNC = 5
  OBJECT = 6
  LAMBDA = 7

# Represents a value, which has a type and its value
class Value:
  def __init__(self, type, value = None):
    self.t = type
    self.v = value

  def value(self):
    return self.v

  def set(self, other):
    self.t = other.t
    self.v = other.v

  def type(self):
    return self.t

# Main interpreter class
class Interpreter(InterpreterBase):
  def __init__(self, console_output=True, input=None, trace_output=False):
    super().__init__(console_output, input)
    self._setup_operations()  # setup all valid binary operations and the types they work on
    self._setup_default_values()  # setup the default values for each type 
    self.trace_output = trace_output

  # run a program, provided in an array of strings, one string per line of source code
  def run(self, program):
    self.program = program
    self._compute_indentation(program)  # determine indentation of every line
    self.tokenized_program = Tokenizer.tokenize_program(program)
    self.func_manager = FunctionManager(self.tokenized_program)
    self._def_functions() #maping of functions
    self.ip = self.func_manager.get_function_info(InterpreterBase.MAIN_FUNC).start_ip
    self.return_stack = []
    self.captures = []
    self.func_to_capture_mapping = {}
    self.terminate = False
    self.env_manager = EnvironmentManager()   # used to track variables/scope
    self.cur_captures = {}

    # main interpreter run loop
    while not self.terminate:
      self._process_line()

  def _process_line(self):
    if self.trace_output:
      print(f"{self.ip:04}: {self.program[self.ip].rstrip()}")
    tokens = self.tokenized_program[self.ip]
    if not tokens:
      self._blank_line()
      return

    args = tokens[1:]

    match tokens[0]:
      case InterpreterBase.ASSIGN_DEF:
        self._assign(args)
      case InterpreterBase.FUNCCALL_DEF:
        self._funccall(args)
      case InterpreterBase.ENDFUNC_DEF:
        self._endfunc()
      case InterpreterBase.IF_DEF:
        self._if(args)
      case InterpreterBase.ELSE_DEF:
        self._else()
      case InterpreterBase.ENDIF_DEF:
        self._endif()
      case InterpreterBase.RETURN_DEF:
        self._return(args)
      case InterpreterBase.WHILE_DEF:
        self._while(args)
      case InterpreterBase.ENDWHILE_DEF:
        self._endwhile(args)
      case InterpreterBase.VAR_DEF: 
        self._define_var(args)
      case InterpreterBase.LAMBDA_DEF:
        self._lambda()
      case InterpreterBase.ENDLAMBDA_DEF:
        self._endlambda()
      case default:
        raise Exception(f'Unknown command: {tokens[0]}')

  def _blank_line(self):
    self._advance_to_next_statement()
  
  def _remove_block_funcs(self):
    cur_env = self.env_manager.environment[-1][-1]
    env_funcs = []
    for el in cur_env:
      if cur_env[el].type() == Type.FUNC:
        env_funcs += el


    for func in self.existing_functions.copy(): #remove the block functions from list of env functions
      if func in env_funcs:
        del self.existing_functions[func]

  def _assign(self, tokens):
   if len(tokens) < 2:
     super().error(ErrorType.SYNTAX_ERROR,"Invalid assignment statement")
   vname = tokens[0]
   has_attr = False
   
   
  # existing_value_type = self._get_value(tokens[0])
   if tokens[0].find('.') != -1:
    self._handle_attr_assignment(tokens)
    return

   existing_value_type = self._get_value(tokens[0])
   value_type = self._eval_expression(tokens[1:])

   if existing_value_type.type() != value_type.type():
     super().error(ErrorType.TYPE_ERROR,
                   f"Trying to assign a variable of {existing_value_type.type()} to a value of {value_type.type()}",
                   self.ip)
   if existing_value_type.type() == Type.OBJECT: #
    self.env_manager.set(tokens[0], value_type)
    self._copy_captured_env(tokens[1], tokens[0])
   else:
    self._set_value(tokens[0], value_type)
   if value_type.type() == Type.FUNC and value_type.value().find(InterpreterBase.LAMBDA_DEF) != -1:
    if self._is_object(tokens[1]):
      obj, method  = tokens[1].split('.')
      self.func_to_capture_mapping.update({tokens[0]: copy.deepcopy(self.func_to_capture_mapping[obj][method])}) #assigning to object method
    else:
      self.func_to_capture_mapping.update({tokens[0] : copy.deepcopy(self.func_to_capture_mapping[tokens[1]])})
   self._advance_to_next_statement()

  
  def _handle_attr_assignment(self, tokens):
    value_type = self._eval_expression(tokens[1:])
    if tokens[0].find('.') != -1:
      obj, attr = tokens[0].split('.')
      existing_value_type = self.env_manager.get(obj)
      if existing_value_type == None:
        super().error(ErrorType.NAME_ERROR, f"Invalid object {obj}", self.ip)
      if existing_value_type.type() != Type.OBJECT:
        super().error(ErrorType.TYPE_ERROR, f"{existing_value_type.type()} does not support \".\" operator", self.ip)

      if value_type.type() == Type.FUNC and value_type.value().find(InterpreterBase.LAMBDA_DEF) != -1:
        my_id = self._get_object_id(obj)

        if my_id not in self.func_to_capture_mapping: 
          self.func_to_capture_mapping.update({my_id: {}})
        self.func_to_capture_mapping[my_id].update({attr : copy.deepcopy(self.func_to_capture_mapping[tokens[1]])})

      
      existing_value_type.v[attr] = value_type

    self._advance_to_next_statement()
  
  def _is_object(self, var):
    if var.find('.') != -1:
      return True
    return False
  def _get_object_id(self, obj):
    my_id = id(self.env_manager.get(obj))
    return my_id
  
  def _copy_captured_env(self, source, target):

    if source not in self.func_to_capture_mapping:
      return
    else: #source object has methods associated with lambda
      if target in self.func_to_capture_mapping:
        objs_to_change = []

        for el in self.func_to_capture_mapping:
          if id(self.func_to_capture_mapping[el]) == id(self.func_to_capture_mapping[target]):
            objs_to_change.append(el)
        
        for obj in objs_to_change:
          self.func_to_capture_mapping.update({obj: self.func_to_capture_mapping[source]})
      else:
        self.func_to_capture_mapping.update({target: self.func_to_capture_mapping[source]})

   
  
  def _check_valid_function(self, funcname):
    if funcname not in self.func_manager.func_cache:
      return False
    
    return True

  def _funccall(self, args):
    is_object_call = False
    this_object = None
    need_capture = False
    obj_name = None
    capture_vars = None
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing function name to call", self.ip)
    if args[0] == InterpreterBase.PRINT_DEF:
      self._print(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.INPUT_DEF:
      self._input(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.STRTOINT_DEF:
      self._strtoint(args[1:])
      self._advance_to_next_statement()

    else:
      arg = args[0]
      self.return_stack.append(self.ip+1)
      if args[0] in self.existing_functions:
        func_name = args[0]
      else:
        if args[0].find('.') != -1: #object 
          obj, attr = args[0].split('.')
          func_name  = self._get_function_name_object(obj, attr)
          obj_name = obj
          capture_vars = self._capture_lambda_object(obj, attr)   
          self.cur_captures = capture_vars     
          this_object = self.env_manager.get(obj)
          is_object_call = True
        else:

          func_name = self._get_function_name(args[0])
          if func_name.find(InterpreterBase.LAMBDA_DEF) != -1:
            capture_vars = self._capture_lambda(args[0])
            self.cur_captures = capture_vars
      self._create_new_environment(func_name, args[1:], is_object_call, this_object, capture_vars, obj_name)  # Create new environment, copy args into new env
      
      self.ip = self._find_first_instruction(func_name)
  
  def _is_lambda(self, funcname):
    if funcname.find(InterpreterBase.LAMBDA_DEF) != -1:
      return True
    real_func = self.env_manager.get(funcname).value()
    if real_func.find(InterpreterBase.LAMBDA_DEF) != -1:
      return True
    return False

  def _capture_lambda(self,func_name):
    if func_name not in self.func_to_capture_mapping:
      return None
    else:
      return self.func_to_capture_mapping[func_name]

  def _capture_lambda_object(self, obj, func_name): 
    my_id = self._get_object_id(obj)
    if my_id not in self.func_to_capture_mapping: #object does not have method associated with lambda
      return None
    elif func_name not in self.func_to_capture_mapping[my_id]: #function is not of type lambda
      return None
    else:
      return self.func_to_capture_mapping[my_id][func_name]

  
  def _lambda(self):
    self._set_result(Value(Type.FUNC, InterpreterBase.LAMBDA_DEF + ":" + str(self.ip)))
    self._capture_vars()
    scope = self.indents[self.ip]
    for line_num in range(self.ip + 1, len(self.tokenized_program)):
      if not self.tokenized_program[line_num]:
        continue
      if self.tokenized_program[line_num][0] == InterpreterBase.ENDLAMBDA_DEF and self.indents[line_num] == scope:
        self.ip = line_num + 1
        return

  def _capture_vars(self):
    self.func_to_capture_mapping['resultf'] = self.env_manager.capture_env_vars()

  def _extract_resultf(self):
    func = self.env_manager.get(InterpreterBase.RESULT_DEF+self.type_to_result[Type.FUNC]).value()
    return func
  
  def _endlambda(self):
    self._endfunc()
    return

  def _get_function_name(self, func):
    func_name = self.env_manager.get(func)
    if func_name == None:
      super().error(ErrorType.NAME_ERROR, f"Invalid function call", self.ip)
    if func_name.type() != Type.FUNC:
      super().error(ErrorType.TYPE_ERROR, f"Invalid function call", self.ip)
    return func_name.value()
  
  def _get_function_name_object(self, obj, attr):
    func_name = self.env_manager.get(obj)
    if func_name == None:
      super().error(ErrorType.NAME_ERROR, f"Invalid funcion call", self.ip)
    if func_name.type() != Type.OBJECT:
      super().error(ErrorType.TYPE_ERROR, f"Invalid function call", self.ip)
    if attr not in func_name.value(): 
      super().error(ErrorType.NAME_ERROR, f"Invalid attribute", self.ip)
    if func_name.value()[attr].type() != Type.FUNC:
      super().error(ErrorType.TYPE_ERROR, f"Invalid function call", self.ip)
    return func_name.value()[attr].value()

  def _get_attr(self, obj_attr):
    obj, attr = obj_attr.split('.')
    return attr

  def _create_new_environment(self, funcname, args, object_call = False, object = None, captured_vars = None, obj_name = None):
    formal_params = self.func_manager.get_function_info(funcname)
    if formal_params is None:
        super().error(ErrorType.NAME_ERROR, f"Unknown function name {funcname}", self.ip)

    if len(formal_params.params) != len(args):
      super().error(ErrorType.NAME_ERROR,f"Mismatched parameter count in call to {funcname}", self.ip)

    tmp_mappings = {}
    for formal, actual in zip(formal_params.params,args):
      formal_name = formal[0]
      formal_typename = formal[1]
      arg = self._get_value(actual)
      if arg.type() != self.compatible_types[formal_typename]:
        super().error(ErrorType.TYPE_ERROR,f"Mismatched parameter type for {formal_name} in call to {funcname}", self.ip)
      if formal_typename in self.reference_types:
        tmp_mappings[formal_name] = arg
      else:
        tmp_mappings[formal_name] = copy.copy(arg)
      if formal_typename == InterpreterBase.FUNC_DEF:
          if not self._is_object(actual) and actual in self.func_to_capture_mapping: #if function captures variable
            self.func_to_capture_mapping[formal_name] = copy.deepcopy(self.func_to_capture_mapping[actual]) # inherit the captures
          elif self._is_object(actual):
            obj, method =  actual.split('.')
            my_id = self._get_object_id(obj)
            if my_id in self.func_to_capture_mapping:
              if method in self.func_to_capture_mapping[my_id]:
                self.func_to_capture_mapping[formal_name] = copy.deepcopy(self.func_to_capture_mapping[my_id][method])

    if object_call: #add this to the list of env parameters
      tmp_mappings[InterpreterBase.THIS_DEF] = object

    # create a new environment for the target function
    # and add our parameters to the env
    self.env_manager.push()

    if captured_vars != None:
      self.env_manager.add_captures_to_env(copy.deepcopy(captured_vars))
    self.env_manager.import_mappings(tmp_mappings)


  def _endfunc(self, return_val = None):
    if not self.return_stack:  # done with main!
      self.terminate = True
    else:
      self.env_manager.pop()  # get rid of environment for the function
      if return_val:
        self._set_result(return_val)
      else:
        # return default value for type if no return value is specified. Last param of True enables
        # creation of result variable even if none exists, or is of a different type
        return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
        if return_type != InterpreterBase.VOID_DEF:
          self._set_result(self.type_to_default[return_type])
      self.ip = self.return_stack.pop()

  def _if(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid if syntax", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean if expression", self.ip)
    if value_type.value():
      self._advance_to_next_statement()
      self.env_manager.block_nest()  # we're in a nested block, so create new env for it
      return
    else:
      for line_num in range(self.ip+1, len(self.tokenized_program)):
        tokens = self.tokenized_program[line_num]
        if not tokens:
          continue
        if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
        if tokens[0] == InterpreterBase.ELSE_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          self.env_manager.block_nest()  # we're in a nested else block, so create new env for it
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _endif(self):
    self._advance_to_next_statement()
    self._remove_block_funcs() # else block
    self.env_manager.block_unnest()

  # we would only run this if we ran the successful if block, and fell into the else at the end of the block
  # so we need to delete the old top environment
  def _else(self):
    self._remove_block_funcs() # funcs in if block
    self.env_manager.block_unnest()   # Get rid of env for block above
    for line_num in range(self.ip+1, len(self.tokenized_program)):
      tokens = self.tokenized_program[line_num]
      if not tokens:
        continue
      if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _return(self,args):
    # do we want to support returns without values?
    return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
    default_value_type = self.type_to_default[return_type]
    if default_value_type.type() == Type.VOID:
      if args:
        super().error(ErrorType.TYPE_ERROR,"Returning value from void function", self.ip)
      self._endfunc()  # no return
      return
    if not args:
      self._endfunc()  # return default value
      return

    #otherwise evaluate the expression and return its value
    value_type = self._eval_expression(args)
    if value_type.type() != default_value_type.type():
      super().error(ErrorType.TYPE_ERROR,"Non-matching return type", self.ip)
    self._endfunc(value_type)

  def _while(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing while expression", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean while expression", self.ip)
    if value_type.value() == False:
      self._exit_while()
      return

    # If true, we advance to the next statement
    self._advance_to_next_statement()
    # And create a new scope
    self.env_manager.block_nest()

  def _exit_while(self):
    while_indent = self.indents[self.ip]
    cur_line = self.ip + 1
    while cur_line < len(self.tokenized_program):
      if not self.tokenized_program[cur_line]:
        cur_line += 1
        pass 
      if self.tokenized_program[cur_line][0] == InterpreterBase.ENDWHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line + 1
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line += 1
    # didn't find endwhile
    super().error(ErrorType.SYNTAX_ERROR,"Missing endwhile", self.ip)

  def _endwhile(self, args):
    # first delete the scope
    self._remove_block_funcs()
    self.env_manager.block_unnest()
    while_indent = self.indents[self.ip]
    cur_line = self.ip - 1
    while cur_line >= 0:
      if self.tokenized_program[cur_line][0] == InterpreterBase.WHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line -= 1
    # didn't find while
    super().error(ErrorType.SYNTAX_ERROR,"Missing while", self.ip)


  def _define_var(self, args):
    if len(args) < 2:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid var definition syntax", self.ip)
    for var_name in args[1:]:
      if self.env_manager.create_new_symbol(var_name) != SymbolResult.OK:
        if var_name not in self.cur_captures:
          super().error(ErrorType.NAME_ERROR,f"Redefinition of variable {args[1]}", self.ip)
      # is the type a valid type?
      if args[0] not in self.type_to_default:
        super().error(ErrorType.TYPE_ERROR,f"Invalid type {args[0]}", self.ip)
      # Create the variable with a copy of the default value for the type
      if args[0] == InterpreterBase.OBJECT_DEF:
        self.env_manager.set(var_name, copy.deepcopy(self.type_to_default[args[0]]))
      else:
       self.env_manager.set(var_name, copy.copy(self.type_to_default[args[0]]))

    self._advance_to_next_statement()

  def _print(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid print call syntax", self.ip)
    out = []
    for arg in args:
      val_type = self._get_value(arg)
      out.append(str(val_type.value()))
    super().output(''.join(out))

  def _input(self, args):
    if args:
      self._print(args)
    result = super().get_input()
    self._set_result(Value(Type.STRING, result))   # return always passed back in result

  def _strtoint(self, args):
    if len(args) != 1:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid strtoint call syntax", self.ip)
    value_type = self._get_value(args[0])
    if value_type.type() != Type.STRING:
      super().error(ErrorType.TYPE_ERROR,"Non-string passed to strtoint", self.ip)
    self._set_result(Value(Type.INT, int(value_type.value())))   # return always passed back in result

  def _advance_to_next_statement(self):
    self.ip += 1
  
  def _def_functions(self):
    self.existing_functions = {}
    for func in self.func_manager.func_cache:
      self.existing_functions[func] = Value(Type.FUNC, func)

  # Set up type-related data structures
  def _setup_default_values(self):
    # set up what value to return as the default value for each type
    self.type_to_default = {}
    self.type_to_default[InterpreterBase.INT_DEF] = Value(Type.INT,0)
    self.type_to_default[InterpreterBase.STRING_DEF] = Value(Type.STRING,'')
    self.type_to_default[InterpreterBase.BOOL_DEF] = Value(Type.BOOL,False)
    self.type_to_default[InterpreterBase.VOID_DEF] = Value(Type.VOID,None)
    self.type_to_default[InterpreterBase.FUNC_DEF] = Value(Type.FUNC, "dummy")
    self.type_to_default[InterpreterBase.OBJECT_DEF] = Value(Type.OBJECT, {})

    # set up what types are compatible with what other types
    self.compatible_types = {}
    self.compatible_types[InterpreterBase.INT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.STRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.BOOL_DEF] = Type.BOOL
    self.compatible_types[InterpreterBase.REFINT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.REFSTRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.REFBOOL_DEF] = Type.BOOL
    self.compatible_types[InterpreterBase.FUNC_DEF] = Type.FUNC
    self.compatible_types[InterpreterBase.LAMBDA_DEF] = Type.FUNC
    self.compatible_types[InterpreterBase.OBJECT_DEF] = Type.OBJECT
    
    self.reference_types = {InterpreterBase.REFINT_DEF, Interpreter.REFSTRING_DEF,
                            Interpreter.REFBOOL_DEF}

    # set up names of result variables: resulti, results, resultb
    self.type_to_result = {}
    self.type_to_result[Type.INT] = 'i'
    self.type_to_result[Type.STRING] = 's'
    self.type_to_result[Type.BOOL] = 'b'
    self.type_to_result[Type.OBJECT] = 'o'
    self.type_to_result[Type.FUNC] = 'f'
    self.type_to_result[Type.LAMBDA] = 'f'

  # run a program, provided in an array of strings, one string per line of source code
  def _setup_operations(self):
    self.binary_op_list = ['+','-','*','/','%','==','!=', '<', '<=', '>', '>=', '&', '|']
    self.binary_ops = {}
    self.binary_ops[Type.INT] = {
     '+': lambda a,b: Value(Type.INT, a.value()+b.value()),
     '-': lambda a,b: Value(Type.INT, a.value()-b.value()),
     '*': lambda a,b: Value(Type.INT, a.value()*b.value()),
     '/': lambda a,b: Value(Type.INT, a.value()//b.value()),  # // for integer ops
     '%': lambda a,b: Value(Type.INT, a.value()%b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.STRING] = {
     '+': lambda a,b: Value(Type.STRING, a.value()+b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.BOOL] = {
     '&': lambda a,b: Value(Type.BOOL, a.value() and b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '|': lambda a,b: Value(Type.BOOL, a.value() or b.value())
    }

  def _compute_indentation(self, program):
    self.indents = [len(line) - len(line.lstrip(' ')) for line in program]

  def _find_first_instruction(self, funcname):
    func_info = self.func_manager.get_function_info(funcname)
    if not func_info:
      super().error(ErrorType.NAME_ERROR,f"Unable to locate {funcname} function")

    return func_info.start_ip

  # given a token name (e.g., x, 17, True, "foo"), give us a Value object associated with it
  def _get_value(self, token):
    if not token:
      super().error(ErrorType.NAME_ERROR,f"Empty token", self.ip)
    if token[0] == '"':
      return Value(Type.STRING, token.strip('"'))
    if token.isdigit() or token[0] == '-':
      return Value(Type.INT, int(token))
    if token == InterpreterBase.TRUE_DEF or token == Interpreter.FALSE_DEF:
      return Value(Type.BOOL, token == InterpreterBase.TRUE_DEF)
    if token.find('.') != -1:
      obj, attr = token.split('.')
      val = self._get_value_of_obj_attr(obj, attr)
      return val
    if token in self.existing_functions: 
      return self.existing_functions[token]

    val = self.env_manager.get(token)
    if val != None:
      return val
    super().error(ErrorType.NAME_ERROR,f"Unknown variable {token}", self.ip)

  def _set_value(self, varname, to_value_type):
    value_type = self.env_manager.get(varname)
    if value_type == None:
      super().error(ErrorType.NAME_ERROR,f"Assignment of unknown variable {varname}", self.ip)
    value_type.set(to_value_type)

  # bind the result[s,i,b] variable in the calling function's scope to the proper Value object
  def _set_result(self, value_type):
    # always stores result in the highest-level block scope for a function, so nested if/while blocks
    # don't each have their own version of result
    result_var = InterpreterBase.RESULT_DEF + self.type_to_result[value_type.type()]
    self.env_manager.create_new_symbol(result_var, True)  # create in top block if it doesn't exist
    self.env_manager.set(result_var, copy.copy(value_type))
  
  def _get_value_of_obj_attr(self, obj, attr):
    
    object =  self.env_manager.get(obj)
    if object == None:
      super().error(ErrorType.NAME_ERROR, f"Object {obj} does not exist.", self.ip)
    if object.type() != Type.OBJECT:
      super().error(ErrorType.TYPE_ERROR, f"Type {object.type()} does not support \".\" operator", self.ip)
    attributes = object.value()
    if attr not in attributes:
      super().error(ErrorType.NAME_ERROR, f"Invalid attribute {attr} for object {obj}", self.ip)
    return attributes[attr] #get the attribute

  # evaluate expressions in prefix notation: + 5 * 6 x
  def _eval_expression(self, tokens):
    stack = []

    for token in reversed(tokens):
      if token in self.binary_op_list:
        v1 = stack.pop()
        v2 = stack.pop()
        if v1.type() != v2.type():
          super().error(ErrorType.TYPE_ERROR,f"Mismatching types {v1.type()} and {v2.type()}", self.ip)
        operations = self.binary_ops[v1.type()]
        if token not in operations:
          super().error(ErrorType.TYPE_ERROR,f"Operator {token} is not compatible with {v1.type()}", self.ip)
        stack.append(operations[token](v1,v2))
      elif token == '!':
        v1 = stack.pop()
        if v1.type() != Type.BOOL:
          super().error(ErrorType.TYPE_ERROR,f"Expecting boolean for ! {v1.type()}", self.ip)
        stack.append(Value(Type.BOOL, not v1.value()))
      else:
        value_type = self._get_value(token)
        stack.append(value_type)

    if len(stack) != 1:
      super().error(ErrorType.SYNTAX_ERROR,f"Invalid expression", self.ip)

    return stack[0]
