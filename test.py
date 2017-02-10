from parsing import Grammar, Rule

rules = [
	Rule('$ROOT', '$Type ?$Type', lambda sems: sems),
	Rule('$Type', '$Person', lambda sems: sems[0]),
	Rule('$Type', '$Song', lambda sems: sems[0]), 		
	Rule('$Person','谢霆锋','谢霆锋'),
	Rule('$Person','谢贤','谢贤'),
	Rule('$Song','歌唱祖国','歌唱祖国'),
	Rule('$Loction','香港','香港'),
	Rule('$Person','人','who'),
	#Rule('$Person','$Loction $Person', lambda sems: (sems[1],"born("+sems[1]+") = " + sems[0])),
	Rule('$Person','$Loction $Person', lambda sems: (sems[0],sems[1])),
	Rule('$Which','哪个','哪个'),
	Rule('$Person','$Which $Person',lambda sems: sems[1]),
	Rule('$Relation', '$FwdRelation', lambda sems: (lambda arg: (sems[0], arg))),
	Rule('$FwdRelation','父亲', '父亲'),
	Rule('$FwdRelation','儿子', '儿子'),
	Rule('$FwdRelation','老公', '老公'),
	Rule('$FwdRelation','歌曲', '歌曲'),
	Rule('$FwdRelation','唱 的', '歌曲'),
	Rule('$De','的', '的'),
	Rule('$Person','谁','who'),
	Rule('$Equal','是', 'Equal'),
	Rule('$Type','$Type $Equal $Type', lambda sems: (sems[1], sems[0], sems[2])),
	#Rule('$Person','$Person $Relation', lambda sems: sems[1](sems[0]) )
	Rule('$Type','$Type ?$De $Relation', lambda sems: sems[2](sems[0]) )
]
grammar = Grammar(rules=rules)
query_list = [
	'谢霆锋 父亲 儿子 是 谁',
	'谁 是 谢霆锋 父亲 儿子',
	'歌唱祖国 是 谁 唱 的',
	'谢霆锋 父亲 儿子 是 谁',
	'谢霆锋 是 谁 的 儿子',
	'谁 是 谢霆锋 的 儿子',
	'哪个 香港 人 的 老公 是 谢贤 的 儿子',
	'歌唱祖国 谁 唱 的'
]
#parses = grammar.parse_input('谢霆锋 父亲 儿子 的 歌曲')
for query in query_list:
	parses = grammar.parse_input(query)

	print("###################################################")
	for parse in parses[:1]:
		print(query + " ==" +str(parse.semantics))