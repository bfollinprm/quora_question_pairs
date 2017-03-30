

#### Exception handling
class Error(Exception):
	''' Base class for exceptions in this module '''
	pass

class InputError(Error):
	''' Exception raised when function is called with incorrect errors in input
	'''
	def __init__(self, expression, message):
		self.expression = expression
		self.message = message

class ColumnError(Error):
	'''
	Exception raides when nonexisting column of DataFrame is referred to
	'''
	def __init__(self, expression, message):
		self.expression = expression
		self.message = message



class InitializeError(Error):
	def __init__(self, expression, message):
		self.expression = expression
		self.message = message
